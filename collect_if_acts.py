"""Collect IF-sourced mean activations for many tasks -- the steering-content counterpart
to the IF head-selection methods.

Writes {task}_{prompt_type}_mean_head_activations_if.pt next to the existing
{task}_{prompt_type}_mean_head_activations.pt, so nothing already cached is touched and
the two can be evaluated against each other with the same head scores.

Stage 1 and the trial set are rebuilt exactly as run_all_tasks.py builds them (same cached
instruction.select, same utils.make_rng(SPLIT_SEED, task, "acts")), so the only difference
between the two activation files is where in the attention the content was read.

Config is the shared conf/ tree, so the flags match run_all_tasks.py:

    python collect_if_acts.py
    python collect_if_acts.py model=llama8b tasks=[antonym,synonym]
    python collect_if_acts.py if_source=auto device=cuda:1

Reusable head scores: lrp / if_patching / attn_diff / frame_cie do not depend on mean
activations. cie and shapley DO (they patch mean_acts into heads), so their cached scores
belong to the last-token activations and should not be paired with these.
"""

import traceback
from datetime import datetime
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import collect
import instruction
import utils
from utils import set_seed


def collect_task(task, model, tokenizer, model_config, prepend_bos, cfg):
    """One task: rebuild stage-1/stage-2 inputs, collect at the IF keys, save. RAISES."""
    short_model_name = cfg.model.name.split("/")[-1]
    save_dir = utils.result_dir(cfg.storage_root, short_model_name, task, cfg.prompt_length)
    root_data_dir = f"{cfg.storage_root}/function_vectors/dataset_files"
    prefixes, separators = dict(cfg.prefixes), dict(cfg.separators)

    if_json = cfg.if_json_auto if cfg.get("if_source", "manual") == "auto" else cfg.if_json_manual
    search_strings = utils.load_json_or_empty(if_json).get(task, {}).get(cfg.model.if_key)
    if not search_strings:
        raise ValueError(f"no {cfg.model.if_key!r} keyword list for task {task!r} in {if_json}")

    # --- same dataset/split as pipeline.run_task (split pinned to utils.SPLIT_SEED) ---
    dataset = utils.load_dataset(task, root_data_dir=root_data_dir,
                                 test_size=cfg.test_split, split_valid=True)

    # --- stage 1: cached, so this is a lookup rather than a recompute ---
    candidate_instructions = instruction.load_candidate_prompts(task, cfg.storage_root)
    instructions, filter_set = instruction.select(
        model, tokenizer, dataset, candidate_instructions, prefixes=prefixes, separators=separators,
        n_best_instructions=cfg.n_best_instructions, prepend_bos=prepend_bos, query_split="train",
        batch_size=cfg.batch_size, save_dir=save_dir, dataset_name=task, force=False)

    min_examples = utils.min_filter_examples(
        cfg.total_activation_examples, cfg.n_cie_trials, cfg.n_best_instructions)
    if len(filter_set) < min_examples:
        raise ValueError(f"Insufficient shared train examples ({len(filter_set)} < {min_examples})")

    # --- stage 2, IF variant: identical trials to pipeline.py, different collection site ---
    trials = instruction.clean_trials(dataset, instructions, cfg.total_activation_examples,
                                      prefixes, separators, prepend_bos, tokenizer,
                                      utils.make_rng(utils.SPLIT_SEED, task, "acts"),
                                      filter_set, "train")

    mean_acts, stats = collect.mean_if_activations(
        model, tokenizer, model_config, trials, search_strings,
        min_hit_rate=cfg.get("if_min_hit_rate", 0.5))

    out_path = Path(save_dir) / f"{task}_{cfg.prompt_type}_mean_head_activations_if.pt"
    utils.save_tensor(out_path, mean_acts)
    logger.info(f"[{task}] saved -> {out_path}")

    return {**stats, "search_strings": list(search_strings), "path": str(out_path)}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("config:\n" + OmegaConf.to_yaml(cfg))
    torch.set_grad_enabled(False)
    set_seed(cfg.seed)

    root_data_dir = f"{cfg.storage_root}/function_vectors/dataset_files"
    if not Path(root_data_dir).exists():
        raise ValueError(f"Dataset directory not found: {root_data_dir}")

    short_model_name = cfg.model.name.split("/")[-1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = Path(cfg.results_dir) / f"{short_model_name}_if_acts_{stamp}.json"

    # output_attentions needs the eager path, whatever the methods list says
    logger.info(f"Loading {cfg.model.name} once for {len(cfg.tasks)} tasks (eager attention)")
    model, tokenizer, model_config = utils.load_gpt_model_and_tokenizer(
        cfg.model.name, device=cfg.device, attn_implementation="eager")
    prepend_bos = not model_config["prepend_bos"]

    summary, failures = {}, {}
    for i, task in enumerate(cfg.tasks, 1):
        logger.info(f"===== [{i}/{len(cfg.tasks)}] {task} =====")
        try:
            set_seed(cfg.seed)
            summary[task] = collect_task(task, model, tokenizer, model_config, prepend_bos, cfg)
        except Exception as e:  # noqa: BLE001 - keep the sweep alive
            failures[task] = f"{type(e).__name__}: {e}"
            logger.warning(f"[{task}] SKIPPED -> {failures[task]}")
            logger.debug(traceback.format_exc())

        utils.save_json(summary_path, {
            "model": short_model_name, "prompt_type": cfg.prompt_type,
            "if_source": cfg.get("if_source", "manual"),
            "total_activation_examples": cfg.total_activation_examples,
            "results": summary, "failures": failures})

    logger.info(f"Done: {len(summary)} ok, {len(failures)} skipped -> {summary_path}")

    print(f"\n{'task':<32}{'hit rate':>10}{'IF tokens':>12}  keywords")
    for task, s in summary.items():
        print(f"{task:<32}{s['hit_rate']:>9.0%}{s['mean_if_tokens']:>12.1f}  {s['search_strings']}")
    for task, reason in failures.items():
        print(f"{task:<32}{'(skipped)':>10}  {reason}")


if __name__ == "__main__":
    main()
