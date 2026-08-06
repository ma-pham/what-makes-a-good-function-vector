"""Run several global-head eval jobs back to back on one GPU, unattended.

``run_global_evals`` is macro-driven (MODELS/SEEDS are module constants), so a
sweep of "these models at these seeds" can only express a rectangle. The runs
still missing are not one -- Qwen needs seed 99 only, 8B needs 42 and 99 -- so
this rebinds those macros per job and calls its main() in one process. One
model load per job, nothing to start by hand between them.

A job that raises is logged and the queue moves on, so a single bad
(model, seed) cannot cost the rest of the night.

    nohup .venv/bin/python -u queue_evals.py > results/eval_queue.log 2>&1 &
"""

import time
import traceback

import run_global_evals as G

DEVICE = "cuda:1"

# (label -> HF id, seeds). Both models already have cie / lrp_auto_if /
# if_patching complete at 41/41 tasks for every seed listed, which is what
# run_global_evals.METHODS asks for under IF_SOURCE="manual".
JOBS = [
    ({"Qwen3-4B": "Qwen/Qwen3-4B-Instruct-2507"}, [99]),
    ({"Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct"}, [42, 99]),
]


def main():
    G.DEVICE = DEVICE
    t_start = time.time()

    for i, (models, seeds) in enumerate(JOBS, 1):
        label = f"{list(models)[0]} seeds {seeds}"
        print(f"\n########## job {i}/{len(JOBS)}: {label} on {DEVICE} ##########", flush=True)
        G.MODELS, G.SEEDS = models, seeds
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
