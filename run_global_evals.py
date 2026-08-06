"""Global (task-averaged) head evals across models and seeds.

Drives eval_all_tasks.py with its module constants overridden, writing one
self-describing results/eval_runs/{model}_{agg}_{prompt}_seed{N}_{ts}.json per
(model, seed) -- see the eval_all_tasks docstring for the format. Head scores,
mean activations and filter sets are all cached, so this runs the steered eval
only -- a few minutes per combination.

No CLI -- edit the macros below and run ``python run_global_evals.py``. Every
macro that is not ``None`` is pushed into eval_all_tasks before the runs start,
so this file is the whole record of what a sweep was.

Combinations whose head scores are incomplete are reported and skipped
(``SKIP_INCOMPLETE``), so this is safe to rerun while a sweep is still filling
them in. Set ``DRY_RUN`` to report readiness and exit.
"""

from __future__ import annotations

from pathlib import Path

import eval_all_tasks as E
import utils

# --- macros -----------------------------------------------------------------
# Label -> HF id. Comment a line out to leave that model out of the sweep.
MODELS = {
    # "Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
    # "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen3-4B": "Qwen/Qwen3-4B-Instruct-2507",
}
SEEDS = [25, 42, 99]                      # which _seed{N}_head_scores.pt to evaluate

DRY_RUN = False                           # report readiness and exit
SKIP_INCOMPLETE = True                    # False: run anyway, on whatever tasks exist

# --- pushed into eval_all_tasks; None keeps that module's own default ---------
DEVICE = "cuda:0"

# Which keyword list IF patching scored with: "manual" reads task_config.json,
# "auto" the extracted one (run_all_tasks' if_source, which tags the artifacts).
# LRP is always the auto variant, so only this one is a choice.
IF_SOURCE = "manual"
IF_METHODS = {"manual": "if_patching", "auto": "if_patching_auto_if"}

# Head-score artifact names. The paper labels come from eval_all_tasks.METHOD_LABELS
# (cie -> AIE, lrp_auto_if -> LRP, if_patching -> IF, if_patching_auto_if -> "IF (auto)").
METHODS = ["cie", "lrp_auto_if", IF_METHODS[IF_SOURCE]]
AGGREGATIONS = ["global"]                 # "global" | "per_task"
STEERINGS = ["dist", "fv"]                # "fv" resolves its edit layer from FV_EDIT_LAYERS
TOPK_HEADS = 20
SCALES = [1.0]
MAX_EXAMPLES = None                       # None = the whole test filter set

PROMPT_TYPE = "instruction"
PROMPT_LENGTH = "short"
STORAGE_ROOT = None                       # None = eval_all_tasks' FV_STORAGE_ROOT default
RESULTS_DIR = None                        # None = results/eval_runs
TASKS = None                              # None = the full list in eval_all_tasks

# Which mean-activation file to steer with: "" is the last-token default, "_if" the
# IF-sourced variant from collect_if_acts.py -- pair that one with lrp*/if_patching
# scores, not cie/shapley (see eval_all_tasks).
MACT_SUFFIX = ""

# {short_model: {method: layer, "default": layer}}; None keeps eval_all_tasks'
# table. Only consulted when "fv" is in STEERINGS.
FV_EDIT_LAYERS = None
# -----------------------------------------------------------------------------

# macro name here -> constant name in eval_all_tasks
OVERRIDES = {
    "DEVICE": "DEVICE",
    "METHODS": "METHODS",
    "AGGREGATIONS": "AGGREGATIONS",
    "STEERINGS": "STEERING_METHODS",
    "TOPK_HEADS": "TOPK_HEADS",
    "SCALES": "EVAL_SCALES",
    "MAX_EXAMPLES": "EVAL_MAX_EXAMPLES",
    "PROMPT_TYPE": "PROMPT_TYPE",
    "PROMPT_LENGTH": "PROMPT_LENGTH",
    "STORAGE_ROOT": "STORAGE_ROOT",
    "RESULTS_DIR": "RESULTS_DIR",
    "TASKS": "TASKS",
    "MACT_SUFFIX": "MACT_SUFFIX",
    "FV_EDIT_LAYERS": "FV_EDIT_LAYERS",
}


def apply_overrides():
    """Pushes every non-None macro into eval_all_tasks, and reports what it set."""
    applied = {}
    for macro, constant in OVERRIDES.items():
        value = globals()[macro]
        if value is None:
            continue
        setattr(E, constant, value)
        applied[constant] = value
    return applied


def head_scores_ready(model_name, method, seed):
    """(present, total) head-score files for this combination."""
    short = model_name[model_name.rfind("/") + 1 :]
    present = sum(
        (Path(utils.result_dir(E.STORAGE_ROOT, short, task, E.PROMPT_LENGTH))
         / f"{task}_{E.PROMPT_TYPE}_{method}_seed{seed}_head_scores.pt").exists()
        for task in E.TASKS
    )
    return present, len(E.TASKS)


def plan():
    """The (model, seed) combinations to run, after reporting what is missing."""
    planned = []
    for label, model_name in MODELS.items():
        short = model_name[model_name.rfind("/") + 1 :]
        if "fv" in E.STEERING_METHODS:  # fail here, not after a model has loaded
            layers = {m: E.resolve_edit_layer(short, m) for m in E.METHODS}
            print(f"      {label:14s} fv edit layers: "
                  + ", ".join(f"{m}=L{l}" for m, l in layers.items()))
        for seed in SEEDS:
            counts = {m: head_scores_ready(model_name, m, seed) for m in E.METHODS}
            incomplete = {m: c for m, c in counts.items() if c[0] < c[1]}
            if incomplete and SKIP_INCOMPLETE:
                detail = ", ".join(f"{m} {c[0]}/{c[1]}" for m, c in incomplete.items())
                print(f"skip  {label:14s} seed {seed:<3d} incomplete head scores: {detail}")
                continue
            note = ""
            if incomplete:
                note = "  (partial: " + ", ".join(f"{m} {c[0]}/{c[1]}" for m, c in incomplete.items()) + ")"
            planned.append((label, model_name, seed))
            print(f"run   {label:14s} seed {seed:<3d} {', '.join(E.METHODS)}{note}")
    return planned


def main():
    applied = apply_overrides()
    print("overrides -> eval_all_tasks:")
    for constant, value in applied.items():
        shown = f"<{len(value)} tasks>" if constant == "TASKS" else value
        print(f"  {constant:20s} {shown}")
    print()

    planned = plan()
    if DRY_RUN or not planned:
        print(f"\n{len(planned)} combination(s) ready; nothing run "
              f"({'DRY_RUN' if DRY_RUN else 'nothing to do'}).")
        return

    for label, model_name, seed in planned:
        print(f"\n===== {label} | {', '.join(E.METHODS)} | {', '.join(E.AGGREGATIONS)} "
              f"| {', '.join(E.STEERING_METHODS)} | seed {seed} =====", flush=True)
        E.MODEL_NAME = model_name
        E.HEAD_SCORES_SEED = seed
        E.main()


if __name__ == "__main__":
    main()
