import os
from pathlib import Path
import random
import json

import torch
from loguru import logger

import utils
from utils import set_seed
import demonstrator
import instruction
import collect
import cie
import transformers.pytorch_utils

import lrp
import eval as steer_eval
import shapley

# The two IF keyword lists ship with the repo, so these resolve regardless of the
# working directory the script is launched from.
REPO_ROOT = Path(__file__).resolve().parent
JSON_PATH_MANUAL = str(REPO_ROOT / "task_config.json")
JSON_PATH_AUTO = str(REPO_ROOT / "auto_formatted_tasks.json")

def load_if_dict(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    logger.warning(f"IF JSON not found at {path}. Returning empty dictionary.")
    return {}

IF_DICT_MANUAL = load_if_dict(JSON_PATH_MANUAL)
IF_DICT_AUTO = load_if_dict(JSON_PATH_AUTO)



def main():
    # --- config ---
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    dataset_name = "antonym"
    prompt_type = "instruction"          # "demonstrator" or "instruction"
    prompt_length = "short"
    device = "cuda:0"
    seed = 42
    test_split = 0.3
    batch_size = 16
    STORAGE_ROOT = os.environ.get("FV_STORAGE_ROOT", "storage")

    prefixes = {"input": "Q:", "output": "A:", "instructions": ""}
    separators = {"input": "\n", "output": "\n\n", "instructions": "\n"}
    total_activation_examples = 100
    force_recompute = False
    force_recompute_heads = True

    attribution_method = "lrp"           # "cie" or "lrp" (demonstrator only)
    lrp_trials_per_prompt = 5            # Will seek exactly 5 successful instances per prompt
    n_best_prompts = 5
    n_cie_trials = lrp_trials_per_prompt * n_best_prompts
    n_shapley_permutations = 1000
    shapley_coalition_size = 4

    steering_method = "dist"             # "fv" or "dist"
    topk_heads = 20
    fv_edit_layer = 11
    eval_scales = [1.0]
    eval_max_examples = None
    use_test_filter = True
    n_icl_examples = 10                  # demonstrator
    n_best_instructions = 5                   # instruction
    instruction_baseline = "equiprobable"  # "empty" or "equiprobable"
    valid_split = True
    # --- end config ---

    # smallest filter set the trial budget can spend on distinct examples; below this
    # the pipeline pseudo-replicates a handful of queries (see run_all_tasks.py)
    min_filter_examples = max(total_activation_examples // n_best_instructions,
                              n_cie_trials // n_best_instructions)
    # Map huggingface model name to your dictionary keys
    if "3.2-3B" in model_name: 
        dict_model_key = "llama3b"
    elif "3.1-8B" in model_name: 
        dict_model_key = "llama8b"
    else: 
        dict_model_key = "qwen4b"

    short_model_name = model_name[model_name.rfind("/") + 1 :]
    root_data_dir = f"{STORAGE_ROOT}/function_vectors/dataset_files"
    save_dir = utils.result_dir(STORAGE_ROOT, short_model_name, dataset_name, prompt_length)

    torch.set_grad_enabled(False)
    set_seed(seed)

    logger.info("Loading model")
    attn_impl = "eager" if attribution_method == "lrp" else None
    model, tokenizer, model_config = utils.load_gpt_model_and_tokenizer(
        model_name, device=device, attn_implementation=attn_impl)
    prepend_bos = not model_config["prepend_bos"]

    logger.info("Loading dataset")
    if not os.path.exists(root_data_dir):
        raise ValueError(f"Dataset directory not found: {root_data_dir}")
    # split pinned to utils.SPLIT_SEED; `seed` drives the stochastic stages only
    dataset = utils.load_dataset(dataset_name, root_data_dir=root_data_dir,
                                 test_size=test_split, split_valid=valid_split)
    logger.debug(f"Loaded {dataset_name}: { {k: len(d) for k, d in dataset.items()} }")

    mact_path = Path(save_dir) / f"{dataset_name}_{prompt_type}_mean_head_activations.pt"
    scores_path = Path(save_dir) / f"{dataset_name}_{prompt_type}_{attribution_method}_head_scores.pt"

    # Determine which dictionary to use based on the method name suffix
    if "lrp_auto_if" == attribution_method:
        formatted_tasks = IF_DICT_AUTO
    else:
        formatted_tasks = IF_DICT_MANUAL

    if prompt_type == "demonstrator":
        filter_set = demonstrator.select(
            model, tokenizer, dataset, prefixes=prefixes, separators=separators,
            n_icl_examples=n_icl_examples, prepend_bos=prepend_bos, query_split="valid",
            save_dir=save_dir, dataset_name=dataset_name, batch_size=batch_size, force=force_recompute)
        test_fs = demonstrator.test_filter_set(
            model, tokenizer, dataset, prefixes=prefixes, separators=separators,
            n_icl_examples=n_icl_examples, prepend_bos=prepend_bos, save_dir=save_dir,
            dataset_name=dataset_name, batch_size=batch_size, force=force_recompute) if use_test_filter else None

        mean_acts = utils.load_or_compute_tensor(mact_path, lambda: collect.mean_activations(
            model, tokenizer, model_config,
            demonstrator.clean_trials(dataset, total_activation_examples, n_icl_examples,
                                      prefixes, separators, prepend_bos, tokenizer, filter_set, "valid"),
            batch_size=batch_size), expected_ndim=3, force=force_recompute)

        if attribution_method == "cie":
            head_scores = utils.load_or_compute_tensor(scores_path, lambda: cie.indirect_effect(
                model, tokenizer, model_config,
                demonstrator.corrupt_trials(dataset, n_cie_trials, n_icl_examples,
                                            prefixes, separators, prepend_bos, tokenizer, filter_set, "valid"),
                mean_acts), expected_ndim=3, force=force_recompute)
        elif attribution_method == "lrp":
            make_trial = lambda: demonstrator.make_clean_trial(
                dataset, n_icl_examples, prefixes, separators, prepend_bos, tokenizer, filter_set, "valid")
            head_scores = utils.load_or_compute_tensor(scores_path, lambda: lrp.attribution(
                model, tokenizer, model_config, make_trial, n_cie_trials, prefixes["output"]),
                expected_ndim=3, force=force_recompute)
        else:
            raise ValueError(f"attribution_method must be 'cie' or 'lrp', got {attribution_method!r}")

    elif prompt_type == "instruction":
        if attribution_method not in ("cie", "shapley", "lrp"):
            raise ValueError(f"instruction branch supports 'cie' or 'shapley' or 'lrp', got {attribution_method!r}")

        candidate_instructions = instruction.load_candidate_prompts(dataset_name, STORAGE_ROOT)

        instructions, filter_set = instruction.select(
            model, tokenizer, dataset, candidate_instructions, prefixes=prefixes, separators=separators,
            n_best_instructions=n_best_instructions, prepend_bos=prepend_bos, query_split="train",
            batch_size=batch_size, save_dir=save_dir, dataset_name=dataset_name, force=force_recompute)
        
        if len(filter_set) < min_filter_examples:
            raise ValueError(f"Insufficient shared train examples for top instructions "
                             f"({len(filter_set)} < {min_filter_examples})")

        test_fs = instruction.test_filter_set(
            model, tokenizer, dataset, instructions, prefixes=prefixes, separators=separators,
            prepend_bos=prepend_bos, save_dir=save_dir, dataset_name=dataset_name,
            batch_size=batch_size, force=force_recompute) if use_test_filter else None

        mean_acts = utils.load_or_compute_tensor(mact_path, lambda: collect.mean_activations(
            model, tokenizer, model_config,
            instruction.clean_trials(dataset, instructions, total_activation_examples,
                                     prefixes, separators, prepend_bos, tokenizer,
                                     # SPLIT_SEED, not seed: the mean-activation cache is seedless by name
                                     utils.make_rng(utils.SPLIT_SEED, dataset_name, "acts"), filter_set, "train"),
            batch_size=batch_size), expected_ndim=3, force=force_recompute)

        if attribution_method == "cie":
            head_scores = utils.load_or_compute_tensor(scores_path, lambda: cie.indirect_effect(
                model, tokenizer, model_config,
                instruction.corrupt_trials(model, tokenizer, dataset, n_cie_trials,
                                           prefixes, separators, prepend_bos,
                                           utils.make_rng(seed, dataset_name, "cie"),
                                           baseline=instruction_baseline,
                                           instructions=instructions, filter_set=filter_set, query_split="train"),
                mean_acts), expected_ndim=3, force=force_recompute)
        elif attribution_method == "lrp":
            # Lookup target strings from the imported config
            search_strings = formatted_tasks.get(dataset_name, {}).get(dict_model_key, [dataset_name])
            
            def lrp_compute_fn():
                trial_generators = []
                for p in instructions:
                    def gen_for_prompt(prompt_string=p):
                        # Convert filter_set to list, shuffle to emulate `random.shuffle(indices_list)`
                        indices = list(filter_set)
                        random.shuffle(indices)
                        split = dataset["train"]
                        for idx in indices:
                            yield utils.word_pairs_to_prompt_data(
                                {"input": [], "output": []}, query_target_pair=split[int(idx)], 
                                instructions=prompt_string, prepend_bos_token=prepend_bos, 
                                shuffle_labels=False, prefixes=prefixes, separators=separators, 
                                tokenizer=tokenizer
                            )
                    trial_generators.append(gen_for_prompt())

                # Pass the generators directly to LRP to handle quota extraction
                target_trials = min(lrp_trials_per_prompt, len(filter_set))
                return lrp.attribution(
                    model, tokenizer, model_config, trial_generators, 
                    target_trials, search_strings, top_k_filter=5
                )

            head_scores = utils.load_or_compute_tensor(
                scores_path, lrp_compute_fn, expected_ndim=3, force=force_recompute_heads
            )
        else:  # shapley
            head_scores = utils.load_or_compute_tensor(scores_path, lambda: shapley.shapley_values(
                model, tokenizer, model_config,
                instruction.corrupt_trials(model, tokenizer, dataset, n_shapley_permutations,
                                           prefixes, separators, prepend_bos,
                                           utils.make_rng(seed, dataset_name, "shapley"),
                                           baseline=instruction_baseline,
                                           instructions=instructions, filter_set=filter_set, query_split="train"),
                mean_acts, utils.make_rng(seed, dataset_name, "shapley", "coal"),
                coalition_size=shapley_coalition_size),
                expected_ndim=3, force=force_recompute)
            
    # elif prompt_type == "demonstrator":
    #     filter_set = demonstrator.select(
    #         model, tokenizer, dataset, prefixes=prefixes, separators=separators,
    #         n_icl_examples=n_icl_examples, prepend_bos=prepend_bos, query_split="valid",
    #         save_dir=save_dir, dataset_name=dataset_name, batch_size=batch_size, force=force_recompute)
    #     test_fs = demonstrator.test_filter_set(
    #         model, tokenizer, dataset, prefixes=prefixes, separators=separators,
    #         n_icl_examples=n_icl_examples, prepend_bos=prepend_bos, save_dir=save_dir,
    #         dataset_name=dataset_name, batch_size=batch_size, force=force_recompute) if use_test_filter else None

    #     mean_acts = utils.load_or_compute_tensor(mact_path, lambda: collect.mean_activations(
    #         model, tokenizer, model_config,
    #         demonstrator.clean_trials(dataset, total_activation_examples, n_icl_examples,
    #                                   prefixes, separators, prepend_bos, tokenizer, filter_set, "valid"),
    #         batch_size=batch_size), expected_ndim=3, force=force_recompute)

    #     if attribution_method == "cie":
    #         head_scores = utils.load_or_compute_tensor(scores_path, lambda: cie.indirect_effect(
    #             model, tokenizer, model_config,
    #             demonstrator.corrupt_trials(dataset, n_cie_trials, n_icl_examples,
    #                                         prefixes, separators, prepend_bos, tokenizer, filter_set, "valid"),
    #             mean_acts), expected_ndim=3, force=force_recompute)
    #     elif attribution_method == "lrp":
    #         make_trial = lambda: demonstrator.make_clean_trial(
    #             dataset, n_icl_examples, prefixes, separators, prepend_bos, tokenizer, filter_set, "valid")
    #         head_scores = utils.load_or_compute_tensor(scores_path, lambda: lrp.attribution(
    #             model, tokenizer, model_config, make_trial, n_cie_trials, prefixes["output"]),
    #             expected_ndim=3, force=force_recompute)
    #     else:
    #         raise ValueError(f"attribution_method must be 'cie' or 'lrp', got {attribution_method!r}")

    else:
        raise ValueError(f"prompt_type must be 'demonstrator' or 'instruction', got {prompt_type!r}")

    logger.info(f"mean_activations shape: {tuple(mean_acts.shape)}")
    logger.info(f"{attribution_method} head_scores shape: {tuple(head_scores.shape)}")

    results, top_heads = steer_eval.evaluate(
        model, tokenizer, model_config, dataset, mean_acts, head_scores,
        steering_method=steering_method, topk_heads=topk_heads, edit_layer=fv_edit_layer,
        scales=eval_scales, prefixes=prefixes, separators=separators, prepend_bos=prepend_bos,
        filter_set=test_fs, max_examples=eval_max_examples)
    eval_path = Path(save_dir) / f"{dataset_name}_{prompt_type}_{attribution_method}_{steering_method}_eval.pt"
    torch.save({"results": results, "top_heads": top_heads}, eval_path)
    logger.info(f"Saved eval results -> {eval_path}")
    return mean_acts, head_scores, results


if __name__ == "__main__":
    main()
