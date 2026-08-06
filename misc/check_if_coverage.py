"""Does every selected instruction actually contain an intentional frame?

The keyword lists in auto_formatted_tasks.json / task_config.json are per *task*, but
the attention-based methods look the keywords up in each individual prompt:

    key_idxs = get_substring_indices(prompt_string, tokenizer, search_strings)

If a task's list was derived as the union over prompts, an individual instruction can
end up with no keyword in it. That instruction then contributes an empty IF, so
if_patching/lrp reject its trials -- silently reducing the trial count, or failing the
task outright if every instruction misses.

This checks, per (model, task, instruction), whether at least one keyword occurs, using
the same case-insensitive substring rule as lrp.get_substring_indices. Reads the cached
test_filter_set.json for the 5 selected instructions; no model, no GPU.

    python check_if_coverage.py                      # auto lists, all three models
    python check_if_coverage.py --if-json task_config.json
    python check_if_coverage.py --model-keys llama3b --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "plots"))
from tasks import COMMON_TASKS  # noqa: E402

STORAGE_ROOT = os.environ.get("FV_STORAGE_ROOT", "storage")
PROMPT_TYPE, PROMPT_LENGTH = "instruction", "short"
MODELS = {
    "llama3b": "Llama-3.2-3B-Instruct",
    "llama8b": "Llama-3.1-8B-Instruct",
    "qwen4b": "Qwen3-4B-Instruct-2507",
}


def selected_instructions(model_dir, task):
    """The 5 instructions the pipeline actually steers with, from the cached filter set."""
    p = (Path(STORAGE_ROOT) / "function_vectors" / f"full_results_prompt_based_{PROMPT_LENGTH}"
         / model_dir / task / f"{task}_{PROMPT_TYPE}_test_filter_set.json")
    if not p.exists():
        return None
    return json.load(open(p)).get("instructions", [])


def covered(instruction, keywords):
    """Same rule as get_substring_indices: case-insensitive substring match."""
    low = instruction.lower()
    return [k for k in keywords if k.lower() in low]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--if-json", default="auto_formatted_tasks.json")
    ap.add_argument("--model-keys", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--verbose", action="store_true", help="list every uncovered instruction")
    args = ap.parse_args()

    if_lists = json.load(open(args.if_json))
    print(f"IF source: {args.if_json}\n")

    for key in args.model_keys:
        model_dir = MODELS[key]
        no_list, missing_fs, gaps, total_instr, uncovered_instr, fully_dead = [], [], [], 0, 0, []

        for task in COMMON_TASKS:
            kws = if_lists.get(task, {}).get(key) or []
            if not kws:
                no_list.append(task)
                continue
            instrs = selected_instructions(model_dir, task)
            if instrs is None:
                missing_fs.append(task)
                continue

            hits = [covered(i, kws) for i in instrs]
            total_instr += len(instrs)
            bad = [i for i, h in zip(instrs, hits) if not h]
            uncovered_instr += len(bad)
            if bad:
                gaps.append((task, len(bad), len(instrs), kws, bad))
            if len(bad) == len(instrs):
                fully_dead.append(task)

        print(f"=== {key} ({model_dir})")
        print(f"  tasks with a keyword list : {len(COMMON_TASKS) - len(no_list)}/{len(COMMON_TASKS)}")
        if no_list:
            print(f"    no list               : {no_list}")
        if missing_fs:
            print(f"    no cached filter set  : {missing_fs}")
        print(f"  instructions checked      : {total_instr}")
        print(f"  instructions with NO IF   : {uncovered_instr}"
              f"  ({uncovered_instr / total_instr:.1%})" if total_instr else "")
        print(f"  tasks with >=1 such       : {len(gaps)}")
        if fully_dead:
            print(f"  !! tasks where NO instruction contains the IF: {fully_dead}")
        for task, nbad, n, kws, bad in (gaps if args.verbose else gaps[:5]):
            print(f"    {task}: {nbad}/{n} uncovered | keywords={kws}")
            if args.verbose:
                for b in bad:
                    print(f"        no match: {b!r}")
        if gaps and not args.verbose and len(gaps) > 5:
            print(f"    ... {len(gaps) - 5} more (use --verbose)")
        print()


if __name__ == "__main__":
    main()
