"""One task through the pipeline: select -> mean acts -> head scores -> steered eval.

Shared by both entry points: run_all_tasks.py loops this over many tasks, main.py
calls it once. It RAISES on an unusable task; the caller decides whether that ends
the run (main.py) or is recorded and skipped (run_all_tasks.py).
"""

from pathlib import Path

import torch
from loguru import logger

import utils
import instruction
import collect
import cie
import shapley
import lrp
import eval as steer_eval
import attention_diff
import frame_cie
import if_patching
import lrp_whole_prompt


# methods that score attention onto the instruction's task words; they need the eager
# attention path and a per-model keyword list
# Methods that score attention onto the instruction's task words: they need the eager
# attention path and an intentional frame (IF), i.e. a keyword list. Conceptually
# if_patching is CIE *with* an IF -- same patch-and-measure shape, but it injects what the
# head read off the IF tokens rather than a task-mean activation, and corrupts by removing
# the instruction rather than replacing it with gibberish. The dispatch below stays literal
# (one method name, one computation, one filename) even though the taxonomy is 2-D.
KEYWORD_METHODS = ("lrp", "attn_diff", "frame_cie", "if_patching")

# Second axis: where a keyword method's IF comes from. The tags reproduce the artifact
# names already on disk, so nothing needs renaming:
#   manual -> ""               lrp_seed{N}          if_patching_seed{N}
#   auto   -> "_auto_if"       lrp_auto_if_seed{N}  (was its own method before)
#   none   -> "_whole_prompt"  lrp_whole_prompt_seed{N}
IF_SOURCE_TAG = {"manual": "", "auto": "_auto_if", "none": "_whole_prompt"}


def run_task(task, model, tokenizer, model_config, prepend_bos, cfg) -> dict:
    """select -> mean acts -> {head scores, steered eval} per method, for one task.

    Returns {method: {scale: {"top1":..,"top2":..,"top3":..,"n":..}}}.
    Raises (and writes a failure marker) if the task has too few usable examples.
    """
    short_model_name = cfg.model.name.split("/")[-1]
    save_dir = utils.result_dir(cfg.storage_root, short_model_name, task, cfg.prompt_length)
    root_data_dir = f"{cfg.storage_root}/function_vectors/dataset_files"
    prefixes, separators = dict(cfg.prefixes), dict(cfg.separators)

    # no seed= here on purpose: the split is pinned to utils.SPLIT_SEED so that
    # sweeping cfg.seed varies only the stochastic stages, not the data
    dataset = utils.load_dataset(task, root_data_dir=root_data_dir,
                                 test_size=cfg.test_split, split_valid=True)
    logger.debug(f"[{task}] splits: { {k: len(d) for k, d in dataset.items()} }")

    # --- stage 1: select instructions + eval filter (method-independent) ---
    candidate_instructions = instruction.load_candidate_prompts(task, cfg.storage_root)
    instructions, filter_set = instruction.select(
        model, tokenizer, dataset, candidate_instructions, prefixes=prefixes, separators=separators,
        n_best_instructions=cfg.n_best_instructions, prepend_bos=prepend_bos, query_split="train",
        batch_size=cfg.batch_size, save_dir=save_dir, dataset_name=task, force=cfg.force_recompute)

    min_examples = utils.min_filter_examples(
        cfg.total_activation_examples, cfg.n_cie_trials, cfg.n_best_instructions)
    if len(filter_set) < min_examples:
        reason = f"Insufficient shared train examples for top prompts ({len(filter_set)} < {min_examples})"
        utils.write_failure_marker(save_dir, "shared_train_examples", reason)
        raise ValueError(reason)

    test_fs = instruction.test_filter_set(
        model, tokenizer, dataset, instructions, prefixes=prefixes, separators=separators,
        prepend_bos=prepend_bos, save_dir=save_dir, dataset_name=task,
        batch_size=cfg.batch_size, force=cfg.force_recompute) if cfg.use_test_filter else None

    if test_fs is not None and len(test_fs) == 0:
        reason = "Zero shared test examples for top prompts"
        utils.write_failure_marker(save_dir, "shared_test_examples", reason)
        raise ValueError(reason)

    # --- stage 2: mean activations (method-independent) ---
    # per-stage RNG so each stage's trials are a pure function of (seed, task, stage),
    # independent of whether an upstream stage cache-hit
    mact_path = Path(save_dir) / f"{task}_{cfg.prompt_type}_mean_head_activations.pt"
    mean_acts = utils.load_cached_tensor(mact_path, expected_ndim=3, force=cfg.force_recompute)
    if mean_acts is None:
        trials = instruction.clean_trials(dataset, instructions, cfg.total_activation_examples,
                                          prefixes, separators, prepend_bos, tokenizer,
                                          # SPLIT_SEED, not cfg.seed: this cache has no seed in its name,
                                          # so its contents must not depend on one (see the note below)
                                          utils.make_rng(utils.SPLIT_SEED, task, "acts"), filter_set, "train")
        mean_acts = utils.save_tensor(mact_path, collect.mean_activations(
            model, tokenizer, model_config, trials, batch_size=cfg.batch_size))

    # --- stages 3+4: per attribution method ---
    task_out = {}
    for method in cfg.methods:
        # head scores carry the seed: which heads make a top-k cut of ~near-tied scores
        # is the seed-sensitive step, so different seeds must not collide in the cache.
        # mean_acts deliberately does NOT -- every seed shares one steering vector, so
        # varying only this seed isolates head-selection stability.
        # if_source tags the artifact but is deliberately NOT part of the RNG key below:
        # the variants must draw the same trials, else a manual-vs-auto difference
        # confounds the keyword list with the sample. Only keyword methods are tagged --
        # cie and shapley never read an IF.
        if cfg.if_source not in IF_SOURCE_TAG:
            raise ValueError(f"if_source must be one of {sorted(IF_SOURCE_TAG)}, got {cfg.if_source!r}")
        tag = IF_SOURCE_TAG[cfg.if_source] if method in KEYWORD_METHODS else ""
        if_json = cfg.if_json_auto if cfg.if_source == "auto" else cfg.if_json_manual
        stem = f"{task}_{cfg.prompt_type}_{method}{tag}_seed{cfg.seed}"
        scores_path = Path(save_dir) / f"{stem}_head_scores.pt"

        # cache: load once here, each branch below only computes, save once at the end
        head_scores = utils.load_cached_tensor(scores_path, expected_ndim=3,
                                               force=cfg.force_recompute_heads)
        if head_scores is None:
            if method in KEYWORD_METHODS:
                module = {"lrp": lrp, "attn_diff": attention_diff,
                          "frame_cie": frame_cie, "if_patching": if_patching}[method]
                search_strings = None

                if cfg.if_source == "none":
                    # attribute over the whole prompt instead of the IF tokens; the other
                    # keyword methods have no meaning without an IF (if_patching's very
                    # corruption is removing it), so only lrp has this variant
                    if method != "lrp":
                        raise ValueError(f"if_source='none' (whole prompt) only applies to 'lrp', "
                                         f"not {method!r}")
                    module = lrp_whole_prompt
                    logger.info(f"[{task}][{method}] whole-prompt LRP, no IF")
                else:
                    logger.info(f"[{task}][{method}] loading IF from {if_json} "
                                f"(if_source={cfg.if_source!r}, if_key={cfg.model.if_key!r})")
                    search_strings = utils.load_json_or_empty(if_json).get(task, {}).get(cfg.model.if_key)
                    if not search_strings:
                        raise ValueError(f"no {cfg.model.if_key!r} keyword list for task {task!r} in "
                                         f"{if_json} (if_source={cfg.if_source!r}); "
                                         f"{method} cannot run without an IF")

                generators = instruction.trial_generators(
                    dataset, instructions, filter_set, prefixes, separators, prepend_bos,
                    tokenizer, utils.make_rng(cfg.seed, task, method), "train")

                trials_per_instruction = min(cfg.lrp_trials_per_prompt, len(filter_set))

                head_scores = module.attribution(model, tokenizer, model_config, generators,
                    trials_per_instruction, search_strings, top_k_filter=5)

            elif method == "cie":
                trials = instruction.corrupt_trials(
                    model, tokenizer, dataset, cfg.n_cie_trials, prefixes, separators, prepend_bos,
                    utils.make_rng(cfg.seed, task, "cie"), baseline=cfg.instruction_baseline,
                    instructions=instructions, filter_set=filter_set, query_split="train")
                head_scores = cie.indirect_effect(model, tokenizer, model_config, trials, mean_acts)

            elif method == "shapley":
                trials = instruction.corrupt_trials(
                    model, tokenizer, dataset, cfg.n_shapley_permutations, prefixes, separators,
                    prepend_bos, utils.make_rng(cfg.seed, task, "shapley"),
                    baseline=cfg.instruction_baseline, instructions=instructions,
                    filter_set=filter_set, query_split="train")
                head_scores = shapley.shapley_values(
                    model, tokenizer, model_config, trials, mean_acts,
                    utils.make_rng(cfg.seed, task, "shapley", "coalitions"),
                    coalition_size=cfg.shapley_coalition_size)

            else:
                raise ValueError(f"unknown attribution method {method!r}")

            utils.save_tensor(scores_path, head_scores)

        results, top_heads = steer_eval.evaluate(
            model, tokenizer, model_config, dataset, mean_acts, head_scores,
            steering_method=cfg.steering_method, topk_heads=cfg.topk_heads,
            edit_layer=cfg.fv_edit_layer, scales=list(cfg.eval_scales),
            prefixes=prefixes, separators=separators, prepend_bos=prepend_bos,
            filter_set=test_fs, max_examples=cfg.eval_max_examples)

        eval_path = Path(save_dir) / f"{stem}_{cfg.steering_method}_eval.pt"
        torch.save({"results": results, "top_heads": top_heads}, eval_path)

        task_out[method] = {
            str(scale): {"top1": r["topk"][1], "top2": r["topk"][2],
                         "top3": r["topk"][3], "n": r["n_examples"]}
            for scale, r in results.items()
        }
        for scale, r in results.items():
            t = r["topk"]
            logger.info(f"[{task}][{method}] scale={scale} top1={t[1]:.3f} "
                        f"top2={t[2]:.3f} top3={t[3]:.3f} (n={r['n_examples']})")

    return task_out
