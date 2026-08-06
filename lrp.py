"""LRP attention-relevance attribution -- alternative to CIE (cie.py).

Adapted to match the exact methodology: processes prompts individually and 
searches a shuffled dataset until exactly `trials_per_prompt` successful 
top-K matches are found.
"""

import importlib
import torch
from loguru import logger
from tqdm import tqdm
import utils

_LXT_PATCHED_MODULES = set()

def _apply_lrp_patch(model):
    module_name = type(model).__module__
    if module_name in _LXT_PATCHED_MODULES:
        return
    from lxt.efficient import monkey_patch
    monkey_patch(importlib.import_module(module_name), verbose=False)
    _LXT_PATCHED_MODULES.add(module_name)

def get_substring_indices(text, tokenizer, search_strings):
    """Finds token indices corresponding to any occurrence of the search string(s)."""
    if isinstance(search_strings, str):
        search_strings = [search_strings]

    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoded['offset_mapping']

    key_indices = set()
    text_lower = text.lower()

    for search_string in search_strings:
        search_lower = search_string.lower()
        start_char = 0
        while True:
            start_char = text_lower.find(search_lower, start_char)
            if start_char == -1:
                break
            end_char = start_char + len(search_string)
            for idx, (start, end) in enumerate(offsets):
                if start == end: continue
                if start < end_char and end > start_char:
                    key_indices.add(idx)
            start_char += 1
    return list(key_indices)


def attribution(model, tokenizer, model_config, trial_generators, trials_per_prompt, search_strings, top_k_filter=5):
    """
    Per-head LRP attribution.
    `trial_generators` is a list of iterators (one per prompt).
    Extracts up to `trials_per_prompt` valid samples from each generator.
    """
    attn_impl = getattr(model.config, "_attn_implementation", None)
    if attn_impl not in (None, "eager"):
        raise ValueError("LRP attribution requires eager attention. Reload with attn_implementation='eager'.")
    
    _apply_lrp_patch(model)

    n_layers = model_config["n_layers"]
    n_heads = model_config["n_heads"]
    device = model.device

    for p in model.parameters():
        p.requires_grad_(False)
    embedding_layer = model.get_input_embeddings()

    all_relevances = []
    logger.info(f"Computing LRP attribution: Aiming for {trials_per_prompt} successes per prompt.")

    for prompt_idx, generator in enumerate(trial_generators):
        successful_samples = 0
        
        pbar = tqdm(total=trials_per_prompt, desc=f"LRP Prompt {prompt_idx+1}", leave=False)
        for prompt_data in generator:
            if successful_samples >= trials_per_prompt:
                break  # Reached quota for this prompt

            prompt_string = utils.create_prompt(prompt_data)
            target = prompt_data["query_target"]["output"]
            target = target[0] if isinstance(target, list) else target

            # 1. Exact Target Token ID
            try:
                target_token_id = utils.get_answer_id(prompt_string, target, tokenizer)[0]
            except (IndexError, AssertionError):
                continue

            # 2. Identify instruction tokens
            key_idxs = get_substring_indices(prompt_string, tokenizer, search_strings)
            if not key_idxs:
                continue

            inputs = tokenizer(prompt_string, return_tensors="pt").to(device)

            with torch.enable_grad():
                inputs_embeds = embedding_layer(inputs["input_ids"])
                inputs_embeds.requires_grad_(True)

                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs.get("attention_mask"),
                    use_cache=False,
                    output_attentions=True,
                )
                next_token_logits = outputs.logits[0, -1, :]

                # 3. Filter: Ensure target is in top-K predictions (Soft Ranking)
                token_rank = utils.compute_individual_token_rank(next_token_logits, target_token_id)
                if token_rank > top_k_filter:
                    continue  # Target not in Top-K

                # 4. Backward pass
                for attn_layer in outputs.attentions:
                    attn_layer.retain_grad()

                model.zero_grad(set_to_none=True)
                next_token_logits[target_token_id].backward()

                # 5. Extract and sum relevance
                layer_scores = [
                    (a * a.grad)[0, :, -1, key_idxs].clamp(min=0).sum(dim=-1).float().detach()
                    for a in outputs.attentions
                ]
                all_relevances.append(torch.stack(layer_scores).cpu())

            successful_samples += 1
            pbar.update(1)
            
        pbar.close()

    if not all_relevances:
        raise ValueError("No valid combinations found where the target was in the Top-K predictions.")

    stacked_relevances = torch.stack(all_relevances)
    logger.info(f"LRP finished. Collected {len(stacked_relevances)} total valid trials.")
    return stacked_relevances