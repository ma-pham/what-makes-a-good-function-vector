"""True IF-Patching (Path Patching of the Intentional Frame).

Runs a corrupt (instruction-less) prompt. At the final token, for a specific head, 
it surgically injects the precise vector that the head *would have* produced 
if it had read the clean Intentional Frame. 
"""

import copy
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

def _layer_values(past_key_values, layer):
    """Cached V for `layer`, across transformers' cache APIs.

    <=4.3x returned a tuple of (K, V) per layer; 4.4x-4.5x wrapped it in a Cache
    exposing .key_cache/.value_cache; 5.x uses .layers[i].keys/.values and is not
    subscriptable at all, so the old indexing raises TypeError.
    """
    if hasattr(past_key_values, "layers"):
        return past_key_values.layers[layer].values
    if hasattr(past_key_values, "value_cache"):
        return past_key_values.value_cache[layer]
    return past_key_values[layer][1]


def attribution(model, tokenizer, model_config, trial_generators, trials_per_prompt, search_strings, top_k_filter=5):
    """
    Per-head True IF-Patching attribution.
    Measures how much P(correct) recovers when we inject the clean Intentional Frame's 
    message directly into the head's output during a corrupted forward pass.
    """
    n_layers = model_config["n_layers"]
    n_heads = model_config["n_heads"]
    device = model.device

    all_relevances = []
    logger.info(f"Computing True IF-Patching for {trials_per_prompt} trials per prompt.")

    for prompt_idx, generator in enumerate(trial_generators):
        successful_samples = 0
        
        pbar = tqdm(total=trials_per_prompt, desc=f"IF-Patch Prompt {prompt_idx+1}", leave=False)
        for prompt_data in generator:
            if successful_samples >= trials_per_prompt:
                break 

            clean_prompt_string = utils.create_prompt(prompt_data)
            
            # Create Corrupt Prompt by stripping the instructions entirely
            corr_prompt_data = copy.deepcopy(prompt_data)
            corr_prompt_data["instructions"] = "" 
            corr_prompt_string = utils.create_prompt(corr_prompt_data)

            target = prompt_data["query_target"]["output"]
            target = target[0] if isinstance(target, list) else target

            # 1. Exact Target Token ID
            try:
                target_token_id = utils.get_answer_id(clean_prompt_string, target, tokenizer)[0]
            except (IndexError, AssertionError):
                continue

            # 2. Identify the exact tokens of the Intentional Frame in the clean prompt
            key_idxs = get_substring_indices(clean_prompt_string, tokenizer, search_strings)
            if not key_idxs:
                continue

            clean_inputs = tokenizer(clean_prompt_string, return_tensors="pt").to(device)
            corr_inputs = tokenizer(corr_prompt_string, return_tensors="pt").to(device)

            with torch.no_grad():
                # --- STEP A: Extract the Clean IF Vector ---
                # We need output_attentions to get routing, and use_cache to get the Value vectors
                clean_out = model(**clean_inputs, output_attentions=True, use_cache=True)
                
                clean_logits = clean_out.logits[:, -1, :]
                token_rank = utils.compute_individual_token_rank(clean_logits, target_token_id)
                if token_rank > top_k_filter:
                    continue  

                # Get baseline performance of the corrupted prompt
                baseline_p = torch.softmax(model(**corr_inputs).logits[:, -1, :], dim=-1)[0, target_token_id].item()

                layer_scores = torch.zeros(n_layers, n_heads)

                # Pre-extract KV caches to save time
                attentions = clean_out.attentions # tuple of (batch, n_heads, seq, seq)
                past_key_values = clean_out.past_key_values

                # --- STEP B: Loop over all layers and heads to test them one by one ---
                for layer in range(n_layers):
                    # Get Attention to IF: shape (n_heads, len(key_idxs))
                    alpha = attentions[layer][0, :, -1, key_idxs] 
                    
                    # Get Value Vectors: shape (n_kv_heads, seq, head_dim)
                    V = _layer_values(past_key_values, layer)[0]
                    
                    # Handle Grouped Query Attention (Llama-3 architecture)
                    n_kv = V.shape[0]
                    n_heads_layer = alpha.shape[0]
                    num_groups = n_heads_layer // n_kv
                    V_expanded = V.repeat_interleave(num_groups, dim=0) # shape (n_heads, seq, head_dim)
                    
                    # Values specifically for the IF tokens: shape (n_heads, len(key_idxs), head_dim)
                    V_if = V_expanded[:, key_idxs, :] 
                    
                    # Calculate the precise vector outputted by the IF: Sum(Alpha * V)
                    # Shape: (n_heads, head_dim)
                    Z_clean_IF = (alpha.unsqueeze(-1) * V_if).sum(dim=1) 

                    o_proj = model.model.layers[layer].self_attn.o_proj
                    
                    for head in range(n_heads_layer):
                        head_dim = Z_clean_IF.shape[1]
                        
                        # Extract this head's slice of the output projection matrix
                        W_h = o_proj.weight[:, head*head_dim : (head+1)*head_dim] # (hidden_size, head_dim)
                        Z_h = Z_clean_IF[head] # (head_dim,)
                        
                        # Project the head's IF vector into the residual stream dimension
                        delta_out = torch.matmul(W_h, Z_h) # (hidden_size,)

                        # --- STEP C: Inject the Vector via Hook ---
                        def get_hook(d_out):
                            def hook(module, args, output):
                                # output shape is (batch, seq, hidden_size)
                                new_out = output.clone()
                                # Inject the clean IF vector into the final token
                                new_out[0, -1, :] += d_out
                                return new_out
                            return hook

                        handle = o_proj.register_forward_hook(get_hook(delta_out))
                        
                        try:
                            # Run corrupted prompt WITH the injected message
                            patch_logits = model(**corr_inputs).logits[:, -1, :]
                            patch_p = torch.softmax(patch_logits, dim=-1)[0, target_token_id].item()
                        finally:
                            handle.remove()

                        # Score is how much the probability RECOVERED
                        layer_scores[layer, head] = patch_p - baseline_p
                
                all_relevances.append(layer_scores)

            successful_samples += 1
            pbar.update(1)
            
        pbar.close()

    if not all_relevances:
        raise ValueError("No valid combinations found where the target was in the Top-K predictions.")

    stacked_relevances = torch.stack(all_relevances)
    logger.info(f"IF-Patching finished. Collected {len(stacked_relevances)} total valid trials.")
    return stacked_relevances
