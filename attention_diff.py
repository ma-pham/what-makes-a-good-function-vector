"""Attention Difference attribution -- ablation study to isolate the Intentional Frame.

Measures the difference in raw attention (softmax output) from the final token 
to the instruction tokens between a clean prompt and a corrupted prompt 
(where instructions are replaced by dummy tokens).
"""

import torch
from loguru import logger
from tqdm import tqdm
import utils

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
    Per-head Attention Difference attribution.
    Collects attention weights from the final token to the instruction tokens,
    then subtracts the attention weights obtained when the instruction is ablated.
    """
    device = model.device
    all_relevances = []
    
    # Identify a safe dummy token to replace the instructions. 
    # Replacing directly in the tensor avoids tokenization length shifts.
    dummy_token_id = tokenizer.convert_tokens_to_ids("The")
    if dummy_token_id is None or dummy_token_id == tokenizer.unk_token_id:
        dummy_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    logger.info(f"Computing Attn-Diff attribution: Aiming for {trials_per_prompt} successes per prompt.")

    for prompt_idx, generator in enumerate(trial_generators):
        successful_samples = 0
        
        pbar = tqdm(total=trials_per_prompt, desc=f"AttnDiff Prompt {prompt_idx+1}", leave=False)
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

            with torch.no_grad():
                # 3. Clean Forward Pass
                outputs = model(
                    **inputs,
                    use_cache=False,
                    output_attentions=True,
                )
                next_token_logits = outputs.logits[0, -1, :]

                # Filter: Ensure target is in top-K predictions on the clean prompt
                token_rank = utils.compute_individual_token_rank(next_token_logits, target_token_id)
                if token_rank > top_k_filter:
                    continue  

                clean_attentions = outputs.attentions

                # 4. Corrupted Forward Pass
                # We replace the instruction token IDs with our dummy token.
                # Doing this at the tensor level guarantees the sequence length remains identical,
                # ensuring the attention matrices align perfectly.
                corrupted_input_ids = inputs["input_ids"].clone()
                corrupted_input_ids[0, key_idxs] = dummy_token_id

                corr_outputs = model(
                    input_ids=corrupted_input_ids,
                    attention_mask=inputs.get("attention_mask"),
                    use_cache=False,
                    output_attentions=True,
                )
                corr_attentions = corr_outputs.attentions

                # 5. Extract and subtract attention (Routing Difference)
                layer_scores = []
                for clean_a, corr_a in zip(clean_attentions, corr_attentions):
                    # Shape of a: (batch, num_heads, sequence_length, sequence_length)
                    # We want: batch 0, all heads, last query token (-1), only key_idxs
                    clean_score = clean_a[0, :, -1, key_idxs].sum(dim=-1).float().cpu()
                    corr_score = corr_a[0, :, -1, key_idxs].sum(dim=-1).float().cpu()
                    
                    # Positive diff means the head attends to the frame specifically 
                    # because it contains valid instructions, not just positional bias.
                    layer_scores.append(clean_score - corr_score)

                all_relevances.append(torch.stack(layer_scores))

            successful_samples += 1
            pbar.update(1)
            
        pbar.close()

    if not all_relevances:
        raise ValueError("No valid combinations found where the target was in the Top-K predictions.")

    stacked_relevances = torch.stack(all_relevances)
    logger.info(f"AttnDiff finished. Collected {len(stacked_relevances)} total valid trials.")
    return stacked_relevances

