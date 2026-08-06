"""Exhaustive steered evaluation of the extracted task representations.

Runs the cartesian product of tasks x methods x aggregations (per_task / global
head sets) x steering methods (dist / fv) x scales, from the cached artifacts of
the extraction pipeline. Nothing is recomputed here: mean activations, head
scores and test filter sets all come off disk.

Output
------
One self-describing JSON per run under ``results/eval_runs/``:

    {"schema", "run", "config", "global_heads", "rows", "failures"}

``config`` pins down everything the numbers depend on -- model, prompt type and
length, storage root, head-score seed, top-k, the task list, the scales, the
mean-activation variant and the resolved FV edit layer per method.
``rows`` is flat: one record per (task, method, aggregation,
steering, scale) carrying its own accuracies *and* the head set it steered with,
so a row is interpretable without the surrounding file. ``global_heads`` records
which tasks each global head set was averaged over -- a global set built from 30
of 41 tasks is not the same object as one built from all 41, and that used to be
invisible.

The older ``results/{model}_eval_all_{ts}.json`` files nest
``results[task][method][agg][steering][scale]`` and record only model, prompt
type, top-k, seed and edit layer. They are still readable by the plot scripts
that predate this; new runs are written in the format above only.
"""

import os
import traceback
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger

import utils
import eval as steer_eval

REPO_ROOT = Path(__file__).resolve().parent

# --- config ------------------------------------------------------------------
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
PROMPT_TYPE = "instruction"
PROMPT_LENGTH = "short"
DEVICE = "cuda:0"
# Which _seed{N}_head_scores.pt to evaluate. The eval itself is deterministic --
# eval_steered walks the cached filter set in order, zero-shot, with no sampling --
# so this selects artifacts and has no other effect on the numbers.
HEAD_SCORES_SEED = 42
TEST_SPLIT = 0.3
STORAGE_ROOT = os.environ.get("FV_STORAGE_ROOT", "storage")
RESULTS_DIR = REPO_ROOT / "results" / "eval_runs"

PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": "\n"}

TOPK_HEADS = 20
EVAL_SCALES = [1.0]
EVAL_MAX_EXAMPLES = None

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

# Artifact name on disk -> the name this method goes by in the paper/plots. The
# artifact name is what {task}_{prompt}_{method}[_seed{N}]_head_scores.pt uses;
# both end up in every row, so a relabelling never orphans the data.
METHODS = ["cie", "lrp_auto_if", "if_patching", "lrp_whole_prompt"]
# The auto-IF variants carry the plain label -- lrp_auto_if is *the* LRP we report --
# and the manual ones are marked, so a plot that happens to hold both never merges
# two different methods into one bar.
METHOD_LABELS = {
    "cie": "AIE",
    "lrp": "LRP (manual IF)",
    "lrp_auto_if": "LRP",
    "lrp_whole_prompt": "LRP-whole",
    "if_patching": "IF",
    "if_patching_auto_if": "IF (auto)",
    "shapley": "Shapley",
    "attn_diff": "AttnDiff",
    "frame_cie": "FrameCIE",
}

# "global" = one head set per method, averaged over every task whose head scores
# are present; "per_task" = the task's own top-k.
AGGREGATIONS = ["global"]

STEERING_METHODS = ["dist", "fv"]

# "fv" injects one residual-stream vector at a single layer, and the best layer
# depends on the model and (weakly) on the attribution method. These come from
# the earlier per-task layer sweep, which only covered cie and lrp; everything
# else falls back to the model's "default" until it is swept. The layer actually
# used is recorded on every fv row, so a stale entry here is always traceable.
FV_EDIT_LAYERS = {
    # lrp_whole_prompt is pinned to the cie layer on purpose: it was never swept, and
    # holding the intervention site fixed makes "does the IF help" a question about the
    # head set rather than about which layer each variant happens to prefer.
    "Llama-3.2-3B-Instruct": {"cie": 9, "lrp": 11, "lrp_whole_prompt": 9, "default": 11},
    "Llama-3.1-8B-Instruct": {"cie": 15, "lrp": 12, "lrp_whole_prompt": 15, "default": 12},
    "Qwen3-4B-Instruct-2507": {"cie": 21, "lrp": 19, "lrp_whole_prompt": 21, "default": 19},
}

# Which mean-activation file to steer with: "" is the last-token default written by
# run_all_tasks.py, "_if" the IF-sourced variant from collect_if_acts.py. The head
# scores are unaffected -- but note cie/shapley scores were computed BY patching the
# last-token means, so only lrp*/if_patching/attn_diff/frame_cie scores pair cleanly
# with "_if".
MACT_SUFFIX = ""

SCHEMA = "fv-eval-run/1"
# -----------------------------------------------------------------------------


def method_label(method):
    return METHOD_LABELS.get(method, method.upper())


def rel(path):
    """Path relative to the storage root, for readable logs."""
    try:
        return str(Path(path).relative_to(STORAGE_ROOT))
    except ValueError:
        return str(path)


def task_dir(short_model_name, task):
    return Path(utils.result_dir(STORAGE_ROOT, short_model_name, task, PROMPT_LENGTH))


def head_scores_path(short_model_name, task, method):
    return task_dir(short_model_name, task) / (
        f"{task}_{PROMPT_TYPE}_{method}_seed{HEAD_SCORES_SEED}_head_scores.pt"
    )


def mean_acts_path(short_model_name, task):
    return task_dir(short_model_name, task) / (
        f"{task}_{PROMPT_TYPE}_mean_head_activations{MACT_SUFFIX}.pt"
    )


def filter_set_path(short_model_name, task):
    return task_dir(short_model_name, task) / f"{task}_{PROMPT_TYPE}_test_filter_set.json"


def resolve_edit_layer(short_model_name, method):
    """FV edit layer for this (model, method), or None when no fv run needs one."""
    if "fv" not in STEERING_METHODS:
        return None
    layers = FV_EDIT_LAYERS.get(short_model_name)
    if layers is None:
        raise SystemExit(
            f"No FV_EDIT_LAYERS entry for {short_model_name}. Add one (a 'default' key is "
            f"enough), or drop 'fv' from STEERING_METHODS."
        )
    if method in layers:
        return int(layers[method])
    if "default" in layers:
        return int(layers["default"])
    raise SystemExit(f"No FV edit layer for {short_model_name}/{method} and no 'default'.")


def run_config(short_model_name, edit_layers, model_config=None):
    """Everything the numbers depend on, in one block."""
    return {
        "model": {
            "name": MODEL_NAME,
            "short": short_model_name,
            "device": DEVICE,
            "n_layers": model_config and model_config.get("n_layers"),
            "n_heads": model_config and model_config.get("n_heads"),
            "head_dim": model_config and model_config.get("head_dim"),
            "resid_dim": model_config and model_config.get("resid_dim"),
        },
        "prompt": {
            "type": PROMPT_TYPE,
            "length": PROMPT_LENGTH,
            "prefixes": PREFIXES,
            "separators": SEPARATORS,
        },
        "data": {
            "storage_root": str(STORAGE_ROOT),
            "test_split": TEST_SPLIT,
            "n_tasks": len(TASKS),
            "tasks": list(TASKS),
        },
        "heads": {
            "methods": list(METHODS),
            "labels": {m: method_label(m) for m in METHODS},
            "aggregations": list(AGGREGATIONS),
            "topk": TOPK_HEADS,
            "scores_seed": HEAD_SCORES_SEED,
        },
        "steering": {
            "methods": list(STEERING_METHODS),
            "scales": list(EVAL_SCALES),
            "fv_edit_layer": edit_layers,
            "mact_suffix": MACT_SUFFIX,
            "max_examples": EVAL_MAX_EXAMPLES,
        },
    }


def precompute_global_heads(short_model_name):
    """Top-K heads per method from the task-averaged scores.

    Returns {method: {"top_heads": [(L, H), ...], "tasks": [...], "missing": [...]}}.
    The task list matters: a set averaged over 30 of 41 tasks is a different object
    from one averaged over all 41, so it travels with the heads.
    """
    logger.info(f"Precomputing global heads across all tasks, under {STORAGE_ROOT} ...")
    out = {}
    for method in METHODS:
        scores, used, missing, loaded = [], [], [], []
        for task in TASKS:
            path = head_scores_path(short_model_name, task, method)
            if not path.exists():
                missing.append(task)
                continue
            logger.debug(f"  head scores <- {rel(path)}")
            s = torch.load(path, map_location="cpu")
            while s.dim() > 2:  # reduce any leading trial dimension(s)
                s = s.mean(dim=0)
            scores.append(s)
            used.append(task)
            loaded.append(path)

        top_heads = []
        if scores:
            avg = torch.stack(scores).mean(dim=0)
            top_heads = steer_eval.select_top_heads(avg, TOPK_HEADS)
            # the file name is what distinguishes cie from lrp_auto_if and one seed
            # from another, so log a concrete one rather than the pattern
            logger.info(f"[{method}] global heads from {len(used)}/{len(TASKS)} score "
                        f"files like {rel(loaded[0])}")
            if missing:
                logger.warning(f"[{method}] no scores for {len(missing)} task(s): {', '.join(missing)}")
            logger.info(f"[{method}] global top-{TOPK_HEADS} heads: {top_heads}")
        else:
            logger.warning(f"No {method} scores for ANY task; its global rows will be skipped.")

        out[method] = {
            "top_heads": [[int(L), int(H)] for L, H in top_heads],
            "n_tasks": len(used),
            "tasks": used,
            "missing_tasks": missing,
        }
    return out


def build_steering(sm, mean_acts, top_heads, model, model_config):
    if sm == "dist":
        n_heads, head_dim = model_config["n_heads"], model_config["head_dim"]
        flat = mean_acts.reshape(-1, head_dim)
        return flat[[L * n_heads + H for L, H in top_heads]]
    if sm == "fv":
        return steer_eval.compute_fv(mean_acts, top_heads, model, model_config)
    raise ValueError(f"Unknown steering method: {sm}")


def report(rows):
    """Mean top-1 per (method, aggregation, steering, scale), for the log."""
    if not rows:
        return
    buckets = {}
    for r in rows:
        buckets.setdefault(
            (r["method_label"], r["method"], r["aggregation"], r["steering"], r["scale"]), []
        ).append(r["top1"])
    logger.info("Mean top-1 over tasks:")
    for key in sorted(buckets):
        label, method, agg, sm, scale = key
        vals = buckets[key]
        logger.info(
            f"  {label:<10s} ({method:<12s}) {agg:<8s} {sm:<4s} scale={scale:<4g} "
            f"top1={sum(vals) / len(vals):.4f}  ({len(vals)} tasks)"
        )


def main():
    torch.set_grad_enabled(False)

    short_model_name = MODEL_NAME[MODEL_NAME.rfind("/") + 1 :]
    root_data_dir = f"{STORAGE_ROOT}/function_vectors/dataset_files"
    if not os.path.exists(root_data_dir):
        raise ValueError(f"Dataset directory not found: {root_data_dir}")

    # resolved before the model is loaded, so a missing entry fails in seconds
    edit_layers = {m: resolve_edit_layer(short_model_name, m) for m in METHODS}

    started = datetime.now()
    timestamp = started.strftime("%Y%m%d_%H%M%S")
    out_path = Path(RESULTS_DIR) / (
        f"{short_model_name}_{'-'.join(AGGREGATIONS)}_{PROMPT_TYPE}"
        f"_seed{HEAD_SCORES_SEED}{MACT_SUFFIX}_{timestamp}.json"
    )

    logger.info(f"Loading model {MODEL_NAME} for exhaustive evaluation...")
    model, tokenizer, model_config = utils.load_gpt_model_and_tokenizer(
        MODEL_NAME, device=DEVICE, attn_implementation="eager"
    )
    prepend_bos = not model_config["prepend_bos"]

    global_heads = precompute_global_heads(short_model_name)

    # rows and failures stay the same objects throughout, so the payload below keeps
    # up with them; only "run" is rewritten, at the end
    rows = []
    failures = {}
    run = {
        "id": out_path.stem,
        "started": started.isoformat(timespec="seconds"),
        "finished": None,
        "seconds": None,
        "complete": False,
    }
    payload = {
        "schema": SCHEMA,
        "run": run,
        "config": run_config(short_model_name, edit_layers, model_config),
        "global_heads": global_heads,
        "rows": rows,
        "failures": failures,
    }

    # config on disk from the start, so a crashed run is still identifiable
    utils.save_json(out_path, payload)

    for i, task in enumerate(TASKS, 1):
        logger.info(f"===== [{i}/{len(TASKS)}] {task} =====")

        mact_path = mean_acts_path(short_model_name, task)
        tfs_path = filter_set_path(short_model_name, task)

        if not mact_path.exists():
            failures[task] = f"Missing mean activations: {mact_path}"
            logger.warning(f"[{task}] SKIPPED -> {failures[task]}")
            continue

        try:
            # no seed= here on purpose: the split is pinned to utils.SPLIT_SEED, so the
            # cached filter sets and mean activations stay valid whatever SEED is
            dataset = utils.load_dataset(
                task, root_data_dir=root_data_dir, test_size=TEST_SPLIT, split_valid=True
            )
            test_fs = utils.load_json(tfs_path)["filter_set"] if tfs_path.exists() else None
            mean_acts = torch.load(mact_path, map_location=DEVICE)
            logger.info(f"[{task}] acts <- {rel(mact_path)}")
            logger.info(
                f"[{task}] filter set <- {rel(tfs_path)} ({len(test_fs)} examples)"
                if test_fs is not None else
                f"[{task}] no filter set at {rel(tfs_path)}; using the whole test split"
            )

            for method in METHODS:
                scores_path = head_scores_path(short_model_name, task, method)

                for agg in AGGREGATIONS:
                    if agg == "global":
                        top_heads = [tuple(h) for h in global_heads[method]["top_heads"]]
                        if not top_heads:
                            continue
                    else:
                        if not scores_path.exists():
                            failures[f"{task}/{method}"] = f"Missing head scores: {scores_path}"
                            logger.warning(f"[{task}][{method}] missing head scores; skipping.")
                            continue
                        logger.info(f"[{task}][{method}] head scores <- {rel(scores_path)}")
                        head_scores = torch.load(scores_path, map_location="cpu")
                        top_heads = steer_eval.select_top_heads(head_scores, TOPK_HEADS)

                    for sm in STEERING_METHODS:
                        steering = build_steering(sm, mean_acts, top_heads, model, model_config)
                        edit_layer = edit_layers[method] if sm == "fv" else None

                        for scale in EVAL_SCALES:
                            r = steer_eval.eval_steered(
                                model=model, tokenizer=tokenizer, model_config=model_config,
                                dataset=dataset, steering_method=sm, steering=steering,
                                top_heads=top_heads, edit_layer=edit_layer or 0, scale=scale,
                                prefixes=PREFIXES, separators=SEPARATORS, prepend_bos=prepend_bos,
                                filter_set=test_fs, max_examples=EVAL_MAX_EXAMPLES,
                            )
                            rows.append({
                                "task": task,
                                "method": method,
                                "method_label": method_label(method),
                                "aggregation": agg,
                                "steering": sm,
                                "scale": float(scale),
                                "edit_layer": edit_layer,
                                "head_scores_seed": HEAD_SCORES_SEED,
                                "topk_heads": TOPK_HEADS,
                                "mact_suffix": MACT_SUFFIX,
                                "top1": r["topk"][1],
                                "top2": r["topk"][2],
                                "top3": r["topk"][3],
                                "n": r["n_examples"],
                                "n_filter_set": test_fs and len(test_fs),
                                "top_heads": [[int(L), int(H)] for L, H in top_heads],
                            })
                            logger.info(
                                f"[{task} | {method_label(method)} | {agg} | {sm}"
                                f"{'' if edit_layer is None else f'@L{edit_layer}'} | scale={scale}] "
                                f"top1={r['topk'][1]:.3f} (n={r['n_examples']})"
                            )

        except Exception as e:
            failures[task] = f"{type(e).__name__}: {e}"
            logger.warning(f"[{task}] CRASHED -> {failures[task]}")
            logger.debug(traceback.format_exc())

        utils.save_json(out_path, payload)  # incremental

    finished = datetime.now()
    run.update(
        finished=finished.isoformat(timespec="seconds"),
        seconds=round((finished - started).total_seconds(), 1),
        complete=True,
    )
    utils.save_json(out_path, payload)

    report(rows)
    logger.info(f"\nEvaluation complete: {len(rows)} rows, {len(failures)} failure(s) -> {out_path}")
    return out_path


if __name__ == "__main__":
    main()
