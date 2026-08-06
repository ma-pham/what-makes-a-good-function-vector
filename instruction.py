"""Instruction (prompt-based) branch: select + test_filter_set + trial construction.

"""

import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

import utils


def load_candidate_prompts(dataset_name, storage_root):
    path = Path(storage_root) / "function_vectors" / "prompts" / f"{dataset_name}_prompts.json"
    data = json.load(open(path))
    assert data["dataset_name"] == dataset_name, f"prompt-file dataset mismatch: {data['dataset_name']}"
    return data["prompts"]


# --- selection: rank candidate instructions, pick top-n, shared filter set ---

def select(model, tokenizer, dataset, candidate_instructions, prefixes, separators,
           n_best_instructions, prepend_bos, query_split="train", batch_size=1,
           save_dir=None, dataset_name=None, force=False):
    """Filter the instruction + train set

    - iterates over whole instruction and train set (saves as .json)
    - filters out invalid examples (target is part of instruction)
    - sorts by instructions with highest accuraies
    - picks n_best_instructions
    - generates train set filter, where all samples lead to succesful predictions

    """
    cache_path = None
    if save_dir is not None and dataset_name is not None:
        cache_path = Path(save_dir) / f"{dataset_name}_instruction_per_prompt_results.json"

    split = dataset[query_split]
    rank_lists = _load_per_prompt_results(cache_path, candidate_instructions) if not force else None

    if rank_lists is None:
        logger.info(f"[select] instruction: ranking {len(candidate_instructions)} instructions on '{query_split}'")
        rank_lists = {}

        # iterate over all instructions
        for instruction in tqdm(candidate_instructions, desc="ranking instructions"):
            rank_lists[instruction] = _eval_instruction_over_split(instruction, split, model, tokenizer, prefixes, separators, prepend_bos, batch_size)
        _save_per_prompt_results(cache_path, rank_lists, candidate_instructions)

    # mask (example, instruction) pairs whose answer leaks from the instruction; the
    # instruction itself stays a candidate, only those examples stop counting for it
    targets = [_raw_target(split[i]) for i in range(len(split))]
    masked = 0
    for instr in candidate_instructions:
        leak = [utils.should_skip_prompt(t, instr) for t in targets]
        masked += sum(1 for r, l in zip(rank_lists[instr], leak) if l and not (isinstance(r, float) and np.isnan(r))) # just a counter
        rank_lists[instr] = [np.nan if l else r for r, l in zip(rank_lists[instr], leak)] # replace rank with nan
    if masked:
        logger.info(f"[select] instruction: masked {masked} leaked (example, instruction) pairs")

    # accuracy over admissible examples only, so a leaky instruction gets no credit
    # for answers it merely echoed from its own text
    top1_acc = {instr: utils.compute_top_k_accuracy(_drop_nan(rank_lists[instr]), k=1) for instr in candidate_instructions}
    top_instructions = sorted(candidate_instructions, key=lambda instr: top1_acc[instr], reverse=True)[:n_best_instructions]
    logger.info(f"[select] instruction: kept {len(top_instructions)} instructions, "
                f"mean top-1 {np.mean([top1_acc[instr] for instr in top_instructions]):.4f}")

    # rank sum == 0 <=> every kept instruction ranked the gold token first; a single NaN
    # (leaked pair) poisons the sum, so one leak against any kept instruction excludes the
    # example entirely -- deliberately strict, the example must be cleanly solved
    summed = np.sum([np.asarray(rank_lists[instr], dtype=float) for instr in top_instructions], axis=0)
    filter_set = np.atleast_1d(np.argwhere(summed == 0).squeeze()).astype(int)
    logger.info(f"[select] instruction: {len(filter_set)} shared usable examples")
    return top_instructions, filter_set


def test_filter_set(model, tokenizer, dataset, instructions, prefixes, separators, prepend_bos,
                    save_dir=None, dataset_name=None, batch_size=1, force=False, split="test"):
    """Eval filter: `split` examples ALL selected instructions answer correctly.
    Cached to {ds}_instruction_test_filter_set.json.

    This is the eval set for steering: it asks "of the examples the clean instruction
    could solve, how many can the steering vector alone recover?", so admitting an
    example the instruction only solved by echoing its answer would flatter the eval.

    """
    cache_path = None
    if save_dir is not None and dataset_name is not None:
        cache_path = Path(save_dir) / f"{dataset_name}_instruction_test_filter_set.json"
    split_data = dataset[split]
    if cache_path is not None and cache_path.exists() and not force:
        data = utils.load_json(cache_path)
        if data.get("instructions") != list(instructions):
            logger.info(f"[test_filter] instruction: cached instruction set differs from selection; recomputing ({cache_path})")
        else:
            logger.info(f"[test_filter] instruction: loading cached test filter set from {cache_path}")
            cached = np.asarray(data["filter_set"], dtype=int)
            # re-mask leakage in case the rule tightened since the cache was written
            # (drop-only: a loosened rule needs force=True, see docstring)
            kept = np.asarray([int(i) for i in cached if not any(
                utils.should_skip_prompt(_raw_target(split_data[int(i)]), instr) for instr in instructions)], dtype=int)
            if len(kept) < len(cached):
                logger.info(f"[test_filter] instruction: dropped {len(cached) - len(kept)} leaked examples from cached filter set")
            return kept

    logger.info(f"[test_filter] instruction: filtering '{split}' via {len(instructions)} selected instructions")
    targets = [_raw_target(split_data[i]) for i in range(len(split_data))]
    rank_lists = []
    for instr in instructions:
        r = np.asarray(_eval_instruction_over_split(instr, split_data, model, tokenizer, prefixes, separators, prepend_bos, batch_size), dtype=float)
        # decision-time leakage mask (same as select's); _eval_instruction_over_split no longer skips
        r[np.asarray([utils.should_skip_prompt(t, instr) for t in targets], dtype=bool)] = np.nan
        rank_lists.append(r)
    summed = np.sum(rank_lists, axis=0)
    filter_set = np.atleast_1d(np.argwhere(summed == 0).squeeze()).astype(int)
    logger.info(f"[test_filter] instruction: {len(filter_set)} shared usable {split} examples")

    if cache_path is not None:
        utils.save_json(cache_path, {"filter_set": filter_set.tolist(), "instructions": list(instructions)})
        logger.info(f"[test_filter] instruction: cached -> {cache_path}")
    return filter_set


def _load_per_prompt_results(cache_path, candidate_instructions):
    """Cached ranks, or None to recompute. null (JSON) == skipped example == NaN."""
    if cache_path is None or not cache_path.exists():
        return None
    cached = utils.load_json(cache_path)["clean_rank_list"]
    if set(cached) != set(candidate_instructions):
        logger.info("[select] instruction: cached instruction set differs from candidates; recomputing")
        return None
    logger.info(f"[select] instruction: loading cached per-instruction results from {cache_path}")
    return {instr: [np.nan if r is None else r for r in cached[instr]] for instr in candidate_instructions}


def _save_per_prompt_results(cache_path, rank_lists, candidate_instructions):
    """Persist raw ranks (+ a top-k summary nothing reads back; select recomputes it)."""
    if cache_path is None:
        return
    serial = {instr: [None if (isinstance(r, float) and np.isnan(r)) else int(r) for r in rank_lists[instr]]
              for instr in candidate_instructions}
    clean_topk = {instr: [[k, utils.compute_top_k_accuracy(_drop_nan(rank_lists[instr]), k)] for k in (1, 2, 3)]
                  for instr in candidate_instructions}
    utils.save_json(cache_path, {"clean_rank_list": serial, "clean_topk": clean_topk})
    logger.info(f"[select] instruction: cached per-instruction results -> {cache_path}")


def _eval_instruction_over_split(instruction, split, model, tokenizer, prefixes, separators, prepend_bos, batch_size):
    """Gold-token rank of EVERY split example under one instruction.

    Deliberately predicate-free (store-everything design): leaked pairs are evaluated
    and stored like any other, so cached ranks stay valid under any future version of
    should_skip_prompt. Leakage is enforced only by the decision-time masks in
    select / test_filter_set."""
    sentences, targets = [], []

    # generate instructions first
    for i in range(len(split)):
        prompt_data = utils.word_pairs_to_prompt_data(
            {"input": [], "output": []}, query_target_pair=split[i], instructions=instruction,
            prepend_bos_token=prepend_bos, shuffle_labels=False,
            prefixes=prefixes, separators=separators, tokenizer=tokenizer)

        target = prompt_data["query_target"]["output"]
        target = target[0] if isinstance(target, list) else target

        sentences.append(utils.create_prompt(prompt_data))
        targets.append(target)

    # gold-token ids in two batch tokenizer calls instead of two per example
    target_token_ids = utils.batch_target_token_ids(sentences, targets, tokenizer)

    # then batched forwards; ranks come back already computed (one sync per batch)
    ranks = []
    for start in range(0, len(sentences), batch_size):
        ranks.extend(utils.batch_ranks(sentences[start : start + batch_size],
                                       target_token_ids[start : start + batch_size], model, tokenizer))
    return ranks


def _drop_nan(ranks):
    return [r for r in ranks if not (isinstance(r, float) and np.isnan(r))]


# --- baseline (corruption) generation ---
#
# CIE/Shapley need a corrupt condition: same prompt shape, no task information. The
# instruction branch cannot shuffle labels (there are no demonstrations), so it swaps
# the instruction for a baseline -- "empty" (nothing) or "equiprobable" (gibberish that
# is as surprising to the model as the real instruction was, controlling for the fact
# that an unpredictable prefix by itself perturbs the model's state).


@torch.no_grad()
def _equiprobable_baseline(model, tokenizer, instruction, rng, margin=0.1, margin_inc=0.1):
    """Token-length-matched gibberish instruction with per-position surprisal matching `instruction`.

    Greedy per position: pick a token whose logprob is within `margin` of the real
    token's (widening by `margin_inc` until something qualifies), excluding special
    tokens. Costs one forward pass per token, so this is generated per trial and is the
    dominant cost of building corrupt trials -- worth caching if trial counts grow.

    `rng` (numpy Generator) breaks ties among matching tokens, so the baseline is a
    deterministic function of the caller's generator.
    """
    device = model.device
    input_ids = tokenizer(instruction, return_tensors="pt").to(device).input_ids
    logprobs = torch.log_softmax(model(input_ids=input_ids).logits.float(), dim=-1)

    n_vocab = logprobs.shape[-1]
    allowed = torch.ones(n_vocab, dtype=torch.bool, device=device)
    for idx in tokenizer.get_added_vocab().values():
        if idx < n_vocab:
            allowed[idx] = False

    sample_ids = [input_ids[0, 0].item()]
    sample_logprobs = None
    for i in range(input_ids.shape[1] - 1):
        current = logprobs[0, i, input_ids[0, i]]
        src = logprobs if sample_logprobs is None else sample_logprobs
        matches = torch.empty(0)
        step = margin
        while matches.numel() == 0:
            matches = torch.argwhere(((src[0, i, :] - current).abs() < step) & allowed).squeeze()
            step += margin_inc
        new_id = matches if matches.numel() == 1 else matches[int(rng.integers(matches.shape[0]))]
        sample_ids.append(int(new_id.item()))
        sample_logprobs = torch.log_softmax(
            model(input_ids=torch.tensor([sample_ids], device=device)).logits.float(), dim=-1)

    if sample_ids and sample_ids[0] == tokenizer.bos_token_id:
        sample_ids = sample_ids[1:]
    return tokenizer.decode(sample_ids)


def _make_baseline_instruction(baseline, model, tokenizer, instruction, rng):
    if baseline == "empty":
        return ""
    if baseline == "equiprobable":
        if instruction is None:
            raise ValueError("baseline='equiprobable' needs the selected `instructions` to derive from")
        return _equiprobable_baseline(model, tokenizer, instruction, rng)
    raise ValueError(f"baseline must be 'empty' or 'equiprobable', got {baseline!r}")


# --- trial construction ---

def _raw_target(sample):
    """Target as stored in the dataset ("A"), not the space-prepended instruction form."""
    target = sample["output"]
    return target[0] if isinstance(target, list) else target


def _admissible(split, fs, instruction):
    """Indices in `fs` whose target does not leak into `instruction`. Raises when
    empty instead of retrying forever (a fully-leaked filter set used to hang here).

    Filtering up-front rather than rejection-sampling per draw is what bounds this:
    the old `while True` redrew from an unchanging pool and could never terminate.
    Selection already excludes leaked pairs, so a raise here means the filter set is
    stale or the baseline instruction happens to contain every target.
    """
    ok = np.asarray([int(i) for i in fs
                     if not utils.should_skip_prompt(_raw_target(split[int(i)]), instruction)], dtype=int)
    if len(ok) == 0:
        raise ValueError(
            f"no admissible example: all {len(fs)} filter-set targets appear in the instruction {instruction!r}")
    return ok


def clean_trials(dataset, instructions, n_trials, prefixes, separators, prepend_bos, tokenizer,
                 rng, filter_set=None, query_split="train"):
    """Distribute `n_trials` across kept instructions; each is instruction + a query.

    Per-instruction draws: n_trials // len(instructions) queries for each instruction (the same
    query under two instructions is a different trial). Drawing with replacement, so a
    filter set smaller than n_per_instruction silently duplicates queries -- the caller's
    minimum-filter-set guard exists to prevent that.

    `rng` (numpy Generator) draws the queries, so the trial set is a pure function of the
    caller's generator, independent of RNG-consumption order elsewhere.
    """
    split = dataset[query_split]
    fs = filter_set if filter_set is not None else np.arange(len(split))
    n_per_instruction = max(n_trials // len(instructions), 1)
    trials = []
    for instruction in instructions:
        ok = _admissible(split, fs, instruction)
        for _ in range(n_per_instruction):
            query = split[rng.choice(ok, 1, replace=False)]
            trials.append(utils.word_pairs_to_prompt_data(
                {"input": [], "output": []}, query_target_pair=query, instructions=instruction,
                prepend_bos_token=prepend_bos, shuffle_labels=False,
                prefixes=prefixes, separators=separators, tokenizer=tokenizer))
    return trials


def trial_generators(dataset, instructions, filter_set, prefixes, separators, prepend_bos,
                     tokenizer, rng, query_split="train"):
    """One lazy generator of prompt_data per instruction, over a shuffled filter set.

    Unlike clean_trials' fixed-size list, the keyword-based attribution methods reject
    trials (the instruction's task words must appear, the target must rank in the top-k)
    and keep drawing until a per-instruction quota is met -- so they need a stream they
    can pull from, not a list that might come up short.

    The shuffle is drawn eagerly here, so RNG consumption doesn't depend on when each
    generator is first advanced.
    """
    split = dataset[query_split]
    fs = filter_set if filter_set is not None else np.arange(len(split))
    generators = []
    for instruction in instructions:
        order = [int(i) for i in rng.permutation(np.asarray(fs))]

        def gen(instruction=instruction, order=order):
            for idx in order:
                yield utils.word_pairs_to_prompt_data(
                    {"input": [], "output": []}, query_target_pair=split[idx],
                    instructions=instruction, prepend_bos_token=prepend_bos,
                    shuffle_labels=False, prefixes=prefixes, separators=separators,
                    tokenizer=tokenizer)

        generators.append(gen())
    return generators


def corrupt_trials(model, tokenizer, dataset, n_trials, prefixes, separators, prepend_bos,
                   rng, baseline="empty", instructions=None, filter_set=None, query_split="train"):
    """`n_trials` corrupted trials: baseline instruction ('empty'/'equiprobable') + a query.

    Instructions are cycled as *sources* for the baseline (each equiprobable baseline is
    length/surprisal-matched to one real instruction), giving n_trials // len(instructions)
    trials per source. The leakage check still applies: a freshly generated baseline is
    unpredictable and can happen to contain the answer, which is why it is re-checked
    here rather than trusted from selection.

    `rng` (numpy Generator) drives both the equiprobable tie-breaks and the query draws.
    """
    split = dataset[query_split]
    fs = filter_set if filter_set is not None else np.arange(len(split))
    trials = []
    for t in range(n_trials):
        instruction = instructions[t % len(instructions)] if instructions else None
        baseline_instr = _make_baseline_instruction(baseline, model, tokenizer, instruction, rng)
        ok = _admissible(split, fs, baseline_instr)
        query = split[rng.choice(ok, 1, replace=False)]
        trials.append(utils.word_pairs_to_prompt_data(
            {"input": [], "output": []}, query_target_pair=query, instructions=baseline_instr,
            prepend_bos_token=prepend_bos, shuffle_labels=False,
            prefixes=prefixes, separators=separators, tokenizer=tokenizer))
    return trials
