"""Sweep the instruction pipeline over many tasks and attribution methods.

Layers:
  * run_task()  -- one task, all methods. RAISES on an unusable task.
  * run_sweep() -- the loop: catches per-task failures, writes markers, saves
                   an incremental summary so a crash never loses the sweep.
  * report()    -- final table.
  * main()      -- Hydra entry point: builds cfg, loads the model ONCE, sweeps.

Config lives in conf/ (see conf/config.yaml). Override from the CLI, e.g.
    python run_all_tasks.py model=qwen4b methods=[cie,shapley] device=cuda:1
    python run_all_tasks.py tasks=[antonym] force_recompute_heads=false

Note the model is deliberately loaded once for the whole sweep, so the task loop
stays in-process rather than using Hydra multirun (which would reload it per task).
"""

import traceback
from datetime import datetime
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import utils
from utils import set_seed
import pipeline
from pipeline import KEYWORD_METHODS, run_task


def run_sweep(model, tokenizer, model_config, prepend_bos, cfg) -> tuple[dict, dict]:
    """Run every task, surviving individual failures. Returns (summary, failures)."""
    short_model_name = cfg.model.name.split("/")[-1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = Path(cfg.results_dir) / f"{short_model_name}_{cfg.prompt_type}_sweep_{stamp}.json"

    summary, failures = {}, {}
    for i, task in enumerate(cfg.tasks, 1):
        logger.info(f"===== [{i}/{len(cfg.tasks)}] {task} =====")
        try:
            # reseed per task so a task's result doesn't depend on what ran before it
            set_seed(cfg.seed)
            summary[task] = run_task(task, model, tokenizer, model_config, prepend_bos, cfg)
        except Exception as e:  # noqa: BLE001 - keep the sweep alive
            failures[task] = f"{type(e).__name__}: {e}"
            logger.warning(f"[{task}] SKIPPED -> {failures[task]}")
            logger.debug(traceback.format_exc())

        # incremental save so a crash/interrupt doesn't lose the sweep
        utils.save_json(summary_path, {
            "model": short_model_name, "prompt_type": cfg.prompt_type,
            "steering_method": cfg.steering_method, "methods": list(cfg.methods),
            "seed": cfg.seed, "results": summary, "failures": failures})

    logger.info(f"Done: {len(summary)} ok, {len(failures)} skipped -> {summary_path}")
    return summary, failures


def report(summary, failures, cfg) -> None:
    """Print the final per-task table and per-method means."""
    methods = list(cfg.methods)
    scale = str(cfg.eval_scales[0])
    header = f"{'task':<32}" + "".join(f"{m + ' top1':>16}" for m in methods)
    print("\n" + header)
    print("-" * len(header))
    for task in cfg.tasks:
        if task not in summary:
            print(f"{task:<32}" + f"{'(skipped)':>16}" * len(methods))
            continue
        print(f"{task:<32}" + "".join(f"{summary[task][m][scale]['top1']:>16.3f}" for m in methods))

    for m in methods:
        vals = [summary[t][m][scale]["top1"] for t in summary]
        if vals:
            print(f"\nmean {m} top1 over {len(vals)} tasks: {sum(vals) / len(vals):.3f}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("config:\n" + OmegaConf.to_yaml(cfg))
    torch.set_grad_enabled(False)
    set_seed(cfg.seed)

    root_data_dir = f"{cfg.storage_root}/function_vectors/dataset_files"
    if not Path(root_data_dir).exists():
        raise ValueError(f"Dataset directory not found: {root_data_dir}")

    # gradient/attention-based methods need the eager attention path; plain CIE/Shapley
    # do not, and the default (sdpa/flash) path is faster
    attn_impl = "eager" if any(m in KEYWORD_METHODS for m in cfg.methods) else None
    logger.info(f"Loading {cfg.model.name} once for {len(cfg.tasks)} tasks x "
                f"{len(cfg.methods)} methods (attn_implementation={attn_impl})")
    model, tokenizer, model_config = utils.load_gpt_model_and_tokenizer(
        cfg.model.name, device=cfg.device, attn_implementation=attn_impl)
    prepend_bos = not model_config["prepend_bos"]

    summary, failures = run_sweep(model, tokenizer, model_config, prepend_bos, cfg)
    report(summary, failures, cfg)


if __name__ == "__main__":
    main()
