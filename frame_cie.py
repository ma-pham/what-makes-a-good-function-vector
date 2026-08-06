"""Frame-Restricted CIE (Path-Specific Attention Ablation).

Measures the causal importance of a head *specifically* reading the intentional frame.
For every (layer, head), we surgically alter the 4D attention mask so that ONLY 
that specific head is blocked from attending to the instruction tokens at the 
final position. We then record the drop in P(correct answer).
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

def _get_frame_ablation_hook(target_head, key_idxs, n_heads):
    """forward_pre_hook: intercepts the LlamaAttention forward pass and surgically 
    modifies the 4D attention mask so `target_head` cannot see `key_idxs`."""
    def pre_hook(module, args, kwargs):
        # In HuggingFace, attention_mask is usually passed as a kwarg
        mask = kwargs.get('attention_mask', None)
        is_kwarg = True
        
        # Fallback if passed positionally (hidden_states, attention_mask, ...)
        if mask is None and len(args) > 1:
            mask = args[1]
            is_kwarg = False

        if mask is not None:
            # Mask shape is typically (batch, 1, seq_len, seq_len)
            # We expand it to (batch, n_heads, seq_len, seq_len) to modify just one head
            if mask.shape[1] == 1:
                new_mask = mask.expand(-1, n_heads, -1, -1).clone()
            else:
                new_mask = mask.clone()
            
            # Apply Path-Specific Ablation:
            # -1e4 safely zeroes out the softmax probability without causing fp16 NaNs
            # We target the last sequence position (-1) attending to the frame (key_idxs)
            for idx in key_idxs:
                new_mask[:, target_head, -1, idx] = -1e4
            
            if is_kwarg:
                kwargs['attention_mask'] = new_mask
            else:
                args = (args[0], new_mask) + args[2:]
                
        return args, kwargs
    return pre_hook

def attribution(model, tokenizer, model_config, trial_generators, trials_per_prompt, search_strings, top_k_filter=5):
    """
    Per-head Frame-Restricted Ablation attribution.
    Collects the drop in target token probability when the attention edge 
    between the final token and the instruction tokens is severed.
    """
    n_layers = model_config["n_layers"]
    n_heads = model_config["n_heads"]
    device = model.device

    all_relevances = []
    logger.info(f"Computing Frame-Restricted CIE (Path-Specific Ablation) for {trials_per_prompt} trials per prompt.")

    for prompt_idx, generator in enumerate(trial_generators):
        successful_samples = 0
        
        pbar = tqdm(total=trials_per_prompt, desc=f"Frame-CIE Prompt {prompt_idx+1}", leave=False)
        for prompt_data in generator:
            if successful_samples >= trials_per_prompt:
                break 

            prompt_string = utils.create_prompt(prompt_data)
            target = prompt_data["query_target"]["output"]
            target = target[0] if isinstance(target, list) else target

            # 1. Exact Target Token ID
            try:
                target_token_id = utils.get_answer_id(prompt_string, target, tokenizer)[0]
            except (IndexError, AssertionError):
                continue

            # 2. Identify instruction tokens (The Intentional Frame)
            key_idxs = get_substring_indices(prompt_string, tokenizer, search_strings)
            if not key_idxs:
                continue

            inputs = tokenizer(prompt_string, return_tensors="pt").to(device)

            with torch.no_grad():
                # 3. Clean Forward Pass to get baseline probability
                clean_logits = model(**inputs, use_cache=False).logits[:, -1, :]
                clean_p = torch.softmax(clean_logits, dim=-1)[0, target_token_id].item()

                # Optional: Filter out examples where the model doesn't even know the answer
                token_rank = utils.compute_individual_token_rank(clean_logits, target_token_id)
                if token_rank > top_k_filter:
                    continue  

                layer_scores = torch.zeros(n_layers, n_heads)

                # 4. Path-Specific Ablation Pass (Iterate Layers and Heads)
                for layer in range(n_layers):
                    # Access the specific self-attention module for the layer
                    # Note: Adjust 'model.layers' if using a non-Llama architecture
                    self_attn_module = model.model.layers[layer].self_attn 
                    
                    for head in range(n_heads):
                        hook = _get_frame_ablation_hook(head, key_idxs, n_heads)
                        # register_forward_pre_hook with with_kwargs=True requires PyTorch 2.0+
                        handle = self_attn_module.register_forward_pre_hook(hook, with_kwargs=True)
                        
                        try:
                            ablation_logits = model(**inputs, use_cache=False).logits[:, -1, :]
                            ablation_p = torch.softmax(ablation_logits, dim=-1)[0, target_token_id].item()
                        finally:
                            handle.remove()
                        
                        # Score is how much the probability DROPPED.
                        # Higher drop = higher reliance on the Intentional Frame
                        layer_scores[layer, head] = clean_p - ablation_p

                all_relevances.append(layer_scores)

            successful_samples += 1
            pbar.update(1)
            
        pbar.close()

    if not all_relevances:
        raise ValueError("No valid combinations found where the target was in the Top-K predictions.")

    stacked_relevances = torch.stack(all_relevances)
    logger.info(f"Frame-CIE finished. Collected {len(stacked_relevances)} total valid trials.")
    return stacked_relevances