"""Run a batch of global-head eval jobs back to back on one GPU, unattended.

The general form of ``queue_evals.py`` (which hardcodes the one gap-filling batch
it was written for). ``run_global_evals`` is macro-driven -- MODELS/SEEDS/METHODS
are module constants, so one invocation can only express a rectangle, and it
iterates model-major. This rebinds those macros per (model, seed) and calls its
main() in one process, seed-major: a full seed lands across all models before the
next seed starts. That ordering is the point under a deadline -- an interrupted
queue still leaves one complete seed rather than one complete model.

A job that raises is logged and the queue moves on, so one bad combination cannot
cost the rest of the night.

    # one method, all models, seed 42 first
    nohup .venv/bin/python -u run_eval_queue.py --methods lrp_whole_prompt \
        --device cuda:0 > results/eval_queue_whole_prompt.log 2>&1 &

    # what would run, without loading a model
    .venv/bin/python run_eval_queue.py --methods lrp_whole_prompt --dry-run
"""

import argparse
import time
import traceback
from pathlib import Path

import run_global_evals as G

MODELS = {
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen3-4B": "Qwen/Qwen3-4B-Instruct-2507",
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 25, 99],
                   help="a run order, not a set: the first finishes across all models first")
    p.add_argument("--methods", nargs="+", default=None,
                   help="head-score artifact names; default: run_global_evals.METHODS")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--results-dir", default=None,
                   help="where the per-run JSONs land; default: eval_all_tasks' results/eval_runs")
    p.add_argument("--dry-run", action="store_true", help="report readiness and exit")
    p.add_argument("--allow-partial", action="store_true",
                   help="evaluate a combination even if some tasks have no head scores yet")
    args = p.parse_args()

    G.DEVICE = args.device
    G.DRY_RUN = args.dry_run
    G.SKIP_INCOMPLETE = not args.allow_partial
    if args.methods:
        G.METHODS = args.methods
    if args.results_dir:
        # resolved through symlinks so the path printed here is the one on disk
        G.RESULTS_DIR = Path(args.results_dir).expanduser().resolve()
        G.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"results -> {G.RESULTS_DIR}")

    jobs = [(m, s) for s in args.seeds for m in args.models]  # seed-major
    t_start = time.time()

    for i, (label, seed) in enumerate(jobs, 1):
        print(f"\n########## job {i}/{len(jobs)}: {label} seed {seed} on {args.device} ##########",
              flush=True)
        G.MODELS, G.SEEDS = {label: MODELS[label]}, [seed]
        t0 = time.time()
        try:
            G.main()
        except Exception:
            traceback.print_exc()
            print(f"########## job {i} FAILED after {time.time() - t0:.0f}s, continuing", flush=True)
        else:
            print(f"########## job {i} done in {time.time() - t0:.0f}s", flush=True)

    print(f"\nqueue finished in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
