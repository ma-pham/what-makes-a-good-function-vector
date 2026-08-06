"""Recompute CIE head scores with today's pipeline, writing outside the storage tree.

The head scores written before commit 5bea8b0 ("clean up, add seeding", 24 Jul) used
an unseeded equiprobable baseline: the tie-break among logprob-matched tokens came
from the global torch RNG rather than the stage generator. CIE measures dP between
the clean and that corrupt prompt, so those scores are not comparable with anything
computed since, and are not reproducible even in principle.

This recomputes them through the current path -- instruction.corrupt_trials ->
cie.indirect_effect -- reading the cached selection, filter sets and mean activations
read-only and writing {out}/{task}_cie_seed{seed}.pt. Nothing under STORAGE_ROOT is
modified, so the result can be inspected before it replaces anything.

    python recompute_cie.py --seed 42 --device cuda:1
    python recompute_cie.py --seed 42 --tasks english-german english-french
    python recompute_cie.py --seed 42 --model Llama-3.1-8B --out results/cie_8b
"""

from __future__ import annotations

import os

import argparse
import time
from pathlib import Path

import torch
from loguru import logger

import cie
import instruction
import utils
from plots.tasks import COMMON_TASKS

MODELS = {
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen3-4B": "Qwen/Qwen3-4B-Instruct-2507",
}
STORAGE_ROOT = os.environ.get("FV_STORAGE_ROOT", "storage")
PROMPT_TYPE, PROMPT_LENGTH = "instruction", "short"
N_CIE_TRIALS, N_BEST, BATCH = 25, 5, 4
BASELINE = "equiprobable"
PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": "\n"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42, help="feeds make_rng for the CIE trials")
    p.add_argument("--model", default="Llama-3.2-3B", choices=list(MODELS))
    p.add_argument("--tasks", nargs="*", default=COMMON_TASKS)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out", default=None, help="default: results/cie_rerun_{model}_seed{seed}")
    args = p.parse_args()

    model_name = MODELS[args.model]
    short = model_name[model_name.rfind("/") + 1 :]
    out_dir = Path(args.out or f"results/cie_rerun_{args.model}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    root_data_dir = f"{STORAGE_ROOT}/function_vectors/dataset_files"

    todo = [t for t in args.tasks if not (out_dir / f"{t}_cie_seed{args.seed}.pt").exists()]
    logger.info(f"{short} seed {args.seed}: {len(todo)} of {len(args.tasks)} tasks to compute -> {out_dir}")
    if not todo:
        return

    torch.set_grad_enabled(False)
    model, tokenizer, model_config = utils.load_gpt_model_and_tokenizer(
        model_name, device=args.device, attn_implementation="eager"
    )
    prepend_bos = not model_config["prepend_bos"]

    for i, task in enumerate(todo, 1):
        t0 = time.time()
        save_dir = utils.result_dir(STORAGE_ROOT, short, task, PROMPT_LENGTH)

        # split pinned to utils.SPLIT_SEED; --seed only drives the trial draw
        dataset = utils.load_dataset(task, root_data_dir=root_data_dir, test_size=0.3, split_valid=True)
        candidates = instruction.load_candidate_prompts(task, STORAGE_ROOT)
        instructions, filter_set = instruction.select(
            model, tokenizer, dataset, candidates, prefixes=PREFIXES, separators=SEPARATORS,
            n_best_instructions=N_BEST, prepend_bos=prepend_bos, query_split="train",
            batch_size=BATCH, save_dir=save_dir, dataset_name=task, force=False,
        )
        mean_acts = torch.load(
            Path(save_dir) / f"{task}_{PROMPT_TYPE}_mean_head_activations.pt", map_location=args.device
        )
        trials = instruction.corrupt_trials(
            model, tokenizer, dataset, N_CIE_TRIALS, PREFIXES, SEPARATORS, prepend_bos,
            utils.make_rng(args.seed, task, "cie"), baseline=BASELINE,
            instructions=instructions, filter_set=filter_set, query_split="train",
        )
        scores = cie.indirect_effect(model, tokenizer, model_config, trials, mean_acts)
        torch.save(scores.cpu(), out_dir / f"{task}_cie_seed{args.seed}.pt")
        logger.info(f"[{i}/{len(todo)}] {task}: {tuple(scores.shape)} in {time.time() - t0:.0f}s")

    print(f"\n{len(list(out_dir.glob(f'*_cie_seed{args.seed}.pt')))} files in {out_dir}")


if __name__ == "__main__":
    main()
