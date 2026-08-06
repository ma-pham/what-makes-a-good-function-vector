import json
import os
import random
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger

import instruction
import utils

# --- Configuration -----------------------------------------------------------
TASKS = [
    # abstractive
    "antonym", "capitalize", "capitalize_first_letter", "capitalize_last_letter",
    "country-capital", "country-currency", "english-french", "english-german", "english-spanish",
    "landmark-country", "national_parks",
    "next_item", "park-country", "present-past", "prev_item",
    "product-company", "singular-plural", "synonym", "word_length",
    # extractive
    "alphabetically_last_3",
    "animal_v_object_3", "animal_v_object_5",
    "choose_first_of_3", "choose_first_of_5",
    "choose_last_of_3", "choose_last_of_5",
    "choose_middle_of_3", "choose_middle_of_5",
    "color_v_animal_3", "color_v_animal_5",
    "concept_v_object_3", "concept_v_object_5",
    "conll2003_location", "conll2003_organization", "conll2003_person",
    "fruit_v_animal_3", "fruit_v_animal_5",
    "object_v_concept_3", "object_v_concept_5",
    "verb_v_adjective_3", "verb_v_adjective_5",
]

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE = os.environ.get("FV_DEVICE", "cuda:0")
STORAGE_ROOT = os.environ.get("FV_STORAGE_ROOT", "storage")
QUERY_SPLIT = "train"
TEST_SPLIT = 0.3

# Intentional Frame Config
N_EXAMPLES = 20  # How many examples to evaluate per prompt (higher = more stable acc drops)
TOP_K = 1
MASK_GRANULARITY = "word"
SEED = 42
N_BEST_PROMPTS = 5
BATCH_SIZE = 32

PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": "\n"}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
short_model_name = MODEL_NAME.split("/")[-1]
SUMMARY_PATH = Path("results") / f"{short_model_name}_IF_sweep_{timestamp}.json"
# -----------------------------------------------------------------------------


def sample_split_examples(split_data, n_examples, seed):
    n_examples = min(n_examples, len(split_data)) if n_examples is not None else len(split_data)
    if n_examples == 0:
        return []
    rng = random.Random(seed)
    indices = list(range(len(split_data)))
    sampled = rng.sample(indices, n_examples)
    return [split_data[i] for i in sampled]


def clock():
    """perf_counter after a CUDA sync -- the scoring launches are async, so timing
    them without this measures how fast the kernels were *queued*, not run."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def mask_prompt_word(prompt_text: str, index: int):
    words = prompt_text.split()
    if index < 0 or index >= len(words):
        raise IndexError("word index out of range")
    masked_words = ["_" if idx == index else word for idx, word in enumerate(words)]
    return " ".join(masked_words), words[index]


def mask_prompt_token(prompt_text: str, index: int, tokenizer):
    tokens = tokenizer.tokenize(prompt_text, add_special_tokens=False)
    if index < 0 or index >= len(tokens):
        raise IndexError("token index out of range")
    masked_tokens = ["_" if idx == index else token for idx, token in enumerate(tokens)]
    return tokenizer.convert_tokens_to_string(masked_tokens).strip(), tokens[index]


def build_prompt_string(prompt_text: str, example: dict, tokenizer, prepend_bos):
    prompt_data = utils.word_pairs_to_prompt_data(
        {"input": [], "output": []},
        query_target_pair=example,
        instructions=prompt_text,
        prefixes=PREFIXES,
        separators=SEPARATORS,
        prepend_bos_token=prepend_bos,
        shuffle_labels=False,
        prepend_space=True, # Explicitly enforce space prepending (FV default)
        tokenizer=tokenizer,
    )
    return utils.create_prompt(prompt_data)


@torch.no_grad()
def score_prompts(model, tokenizer, prompts, targets, batch_size=None):
    """Rank of the gold first token per (prompt, target) pair -- same shape as instruction.py."""
    # read BATCH_SIZE at call time, not as a default argument: a default would be bound
    # at import and silently ignore `find_if.BATCH_SIZE = ...` overrides
    batch_size = BATCH_SIZE if batch_size is None else batch_size

    # gold-token ids in two batch tokenizer calls instead of two per example
    target_token_ids = utils.batch_target_token_ids(prompts, targets, tokenizer)

    ranks = []
    for start in range(0, len(prompts), batch_size):
        ranks.extend(utils.batch_ranks(prompts[start : start + batch_size],
                                       target_token_ids[start : start + batch_size], model, tokenizer))
    return ranks


def analyze_prompt(model, tokenizer, prompt_text, examples, top_k, mask_granularity, prepend_bos):
    if mask_granularity == "token":
        tokens = tokenizer.tokenize(prompt_text, add_special_tokens=False)
        mask_fn = lambda idx: mask_prompt_token(prompt_text, idx, tokenizer)
        elements = tokens
    else:
        words = prompt_text.split()
        mask_fn = lambda idx: mask_prompt_word(prompt_text, idx)
        elements = words

    results = []
    t0 = clock()

    # Prepend space to targets to match tokenizer/model expectations
    baseline_prompts = [build_prompt_string(prompt_text, ex, tokenizer, prepend_bos) for ex in examples]
    baseline_targets = [" " + str(ex["output"]) for ex in examples]

    baseline_ranks = score_prompts(model, tokenizer, baseline_prompts, baseline_targets)
    baseline_seconds = clock() - t0
    baseline_correct = sum(1 for rank in baseline_ranks if rank < top_k)
    baseline_accuracy = baseline_correct / max(len(baseline_ranks), 1)

    for idx, elem_text in enumerate(elements):
        masked_prompt_text, masked_element = mask_fn(idx)
        masked_prompts = [build_prompt_string(masked_prompt_text, ex, tokenizer, prepend_bos) for ex in examples]
        masked_ranks = score_prompts(model, tokenizer, masked_prompts, baseline_targets)
        
        masked_correct = sum(1 for rank in masked_ranks if rank < top_k)
        masked_accuracy = masked_correct / max(len(masked_ranks), 1)

        results.append({
            "index": idx,
            "element": elem_text,
            "masked_element": masked_element,
            "masked_prompt": masked_prompt_text,
            "masked_accuracy": masked_accuracy,
            "accuracy_drop": baseline_accuracy - masked_accuracy,
            "baseline_accuracy": baseline_accuracy,
        })

    # everything above is the method itself (mask -> build -> score); the sort and the
    # bookkeeping below are not, so the clock stops here
    prompt_seconds = clock() - t0

    results.sort(key=lambda x: x["accuracy_drop"], reverse=True)

    # Return top 3 elements causing largest drop for logging purposes
    top_3_words = [r["element"] for r in results[:3] if r["accuracy_drop"] > 0]

    return {
        "prompt_text": prompt_text,
        "baseline_accuracy": baseline_accuracy,
        "most_impactful_elements": top_3_words,
        "results": results,
        "timing": {
            "seconds": prompt_seconds,
            "baseline_seconds": baseline_seconds,
            "n_elements": len(elements),
            # one scoring call per element, plus the baseline, each over len(examples)
            "n_forward_examples": (len(elements) + 1) * len(examples),
        },
    }


def run_task(task, model, tokenizer, model_config, prepend_bos):
    """Run select -> evaluate baseline -> evaluate masked IF for a single task."""
    save_dir = utils.result_dir(STORAGE_ROOT, short_model_name, task, "short")
    root_data_dir = f"{STORAGE_ROOT}/function_vectors/dataset_files"

    # 1. Load Dataset
    dataset = utils.load_dataset(task, root_data_dir=root_data_dir, test_size=TEST_SPLIT, split_valid=True)
    
    # 2. Select instructions & filter set
    prompts_path = Path(STORAGE_ROOT) / "function_vectors" / "prompts" / f"{task}_prompts.json"
    if not prompts_path.exists():
        raise FileNotFoundError(f"Missing prompts file: {prompts_path}")
    candidate_prompts = utils.load_json(prompts_path)["prompts"]
    
    prompts, filter_set = instruction.select(
        model, tokenizer, dataset, candidate_prompts, prefixes=PREFIXES, separators=SEPARATORS,
        n_best_instructions=N_BEST_PROMPTS, prepend_bos=prepend_bos, query_split=QUERY_SPLIT,
        batch_size=BATCH_SIZE, save_dir=save_dir, dataset_name=task, force=False)

    if len(filter_set) == 0:
        raise ValueError("Filter set is empty (no shared usable examples for top prompts)")

    # 3. Sample Examples
    split_data = dataset[QUERY_SPLIT]
    example_indices = [int(i) for i in filter_set]
    examples = [split_data[idx] for idx in example_indices]
    examples = sample_split_examples(examples, N_EXAMPLES, SEED)

    # 4. Analyze each prompt -- this loop is the method; steps 1-3 above are dataset
    # loading, cached instruction selection and sampling, and are deliberately outside
    # the clock (as is the incremental save in main)
    task_results = []
    t0 = clock()
    for prompt_idx, prompt_text in enumerate(prompts):
        logger.debug(f"[{task}] Analyzing IF for prompt {prompt_idx + 1}/{len(prompts)}...")
        analysis = analyze_prompt(model, tokenizer, prompt_text, examples, TOP_K, MASK_GRANULARITY, prepend_bos)
        task_results.append(analysis)

        # Quick logging for visibility
        logger.info(f"[{task}] P{prompt_idx+1} Acc: {analysis['baseline_accuracy']:.2f} | "
                    f"{analysis['timing']['seconds']:.2f}s | Key words: {analysis['most_impactful_elements']}")
    task_seconds = clock() - t0

    n_elements = sum(p["timing"]["n_elements"] for p in task_results)
    n_forward_examples = sum(p["timing"]["n_forward_examples"] for p in task_results)
    logger.info(f"[{task}] IF method: {task_seconds:.2f}s over {len(prompts)} prompts / "
                f"{n_elements} {MASK_GRANULARITY}s "
                f"({1000 * task_seconds / max(n_elements, 1):.0f} ms per {MASK_GRANULARITY})")

    return {
        "n_examples": len(examples),
        "prompts": task_results,
        "timing": {
            "seconds": task_seconds,
            "n_prompts": len(prompts),
            "n_elements": n_elements,
            "n_forward_examples": n_forward_examples,
            "seconds_per_element": task_seconds / max(n_elements, 1),
        },
    }


def main():
    torch.set_grad_enabled(False)
    utils.set_seed(SEED)

    root_data_dir = f"{STORAGE_ROOT}/function_vectors/dataset_files"
    if not os.path.exists(root_data_dir):
        raise ValueError(f"Dataset directory not found: {root_data_dir}")

    logger.info(f"Loading model {MODEL_NAME} (once for {len(TASKS)} tasks)")
    model, tokenizer, model_config = utils.load_gpt_model_and_tokenizer(MODEL_NAME, device=DEVICE)
    prepend_bos = not model_config["prepend_bos"]
    model.eval()

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary, failures = {}, {}

    for i, task in enumerate(TASKS, 1):
        logger.info(f"===== [{i}/{len(TASKS)}] {task} =====")
        try:
            # Reseed per task for consistency across independent runs
            utils.set_seed(SEED)
            summary[task] = run_task(task, model, tokenizer, model_config, prepend_bos)
        except Exception as e:
            failures[task] = f"{type(e).__name__}: {e}"
            logger.warning(f"[{task}] SKIPPED -> {failures[task]}")
            logger.debug(traceback.format_exc())

        # Incremental save
        utils.save_json(SUMMARY_PATH, {
            "model": short_model_name,
            "mask_granularity": MASK_GRANULARITY,
            "top_k": TOP_K,
            "n_examples_per_prompt": N_EXAMPLES,
            "batch_size": BATCH_SIZE,
            "device": DEVICE,
            "timing": sweep_timing(summary),
            "results": summary,
            "failures": failures
        })

    logger.info(f"Done: {len(summary)} ok, {len(failures)} skipped -> {SUMMARY_PATH}")

    # Final Summary Print
    print("\n" + "="*50)
    print("SWEEP COMPLETE - TOP IMPACT WORDS PER TASK (P1)")
    print("="*50)
    for task, res in summary.items():
        if res["prompts"]:
            p1 = res["prompts"][0]
            print(f"{task:<25} Acc: {p1['baseline_accuracy']:.2f} | {p1['most_impactful_elements']}")

    report_timing(summary)


def sweep_timing(summary):
    """Method-only totals over the tasks finished so far (no loading, no IO)."""
    per_task = [res["timing"] for res in summary.values() if "timing" in res]
    if not per_task:
        return {}
    seconds = sum(t["seconds"] for t in per_task)
    n_elements = sum(t["n_elements"] for t in per_task)
    n_forward_examples = sum(t["n_forward_examples"] for t in per_task)
    return {
        "seconds": seconds,
        "n_tasks": len(per_task),
        "n_prompts": sum(t["n_prompts"] for t in per_task),
        "n_elements": n_elements,
        "n_forward_examples": n_forward_examples,
        "seconds_per_task": seconds / len(per_task),
        "seconds_per_element": seconds / max(n_elements, 1),
        "examples_per_second": n_forward_examples / max(seconds, 1e-9),
    }


def report_timing(summary):
    """Per-task table + sweep totals for the masking/scoring work alone."""
    total = sweep_timing(summary)
    if not total:
        return

    print("\n" + "=" * 72)
    print(f"IF METHOD TIMING (excludes model load, dataset load, selection, IO)")
    print("=" * 72)
    print(f"{'task':<28}{'seconds':>10}{MASK_GRANULARITY + 's':>10}{'ms/' + MASK_GRANULARITY:>12}")
    for task, res in sorted(summary.items(), key=lambda kv: -kv[1].get("timing", {}).get("seconds", 0)):
        t = res.get("timing")
        if t:
            print(f"{task:<28}{t['seconds']:>10.2f}{t['n_elements']:>10}"
                  f"{1000 * t['seconds_per_element']:>12.0f}")

    print("-" * 72)
    print(f"{'TOTAL':<28}{total['seconds']:>10.2f}{total['n_elements']:>10}"
          f"{1000 * total['seconds_per_element']:>12.0f}")
    print(f"\n{total['n_tasks']} tasks / {total['n_prompts']} prompts / {total['n_elements']} "
          f"{MASK_GRANULARITY}s / {total['n_forward_examples']} scored examples")
    print(f"{total['seconds'] / 60:.1f} min total, {total['seconds_per_task']:.1f} s per task, "
          f"{total['examples_per_second']:.0f} examples/s "
          f"(N_EXAMPLES={N_EXAMPLES}, BATCH_SIZE={BATCH_SIZE}, {DEVICE})")


if __name__ == "__main__":
    main()