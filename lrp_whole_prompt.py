"""LRP attention-relevance attribution over the full prompt.

This variant follows the same LRP procedure as lrp.py, but it does not focus on
an intentional-frame substring. Instead, it aggregates relevance from the final
position to every token in the prompt, so the score reflects the whole prompt's
contribution to the target token.
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


def attribution(model, tokenizer, model_config, trial_generators, trials_per_prompt, search_strings=None, top_k_filter=5):
    """Per-head LRP attribution over the entire prompt sequence.

    The `search_strings` argument is accepted for interface compatibility with the
    other keyword-based attribution methods, but it is ignored here.
    """
    attn_impl = getattr(model.config, "_attn_implementation", None)
    if attn_impl not in (None, "eager"):
        raise ValueError("Whole-prompt LRP attribution requires eager attention. Reload with attn_implementation='eager'.")

    _apply_lrp_patch(model)

    n_layers = model_config["n_layers"]
    n_heads = model_config["n_heads"]
    device = model.device

    for p in model.parameters():
        p.requires_grad_(False)
    embedding_layer = model.get_input_embeddings()

    all_relevances = []
    logger.info(f"Computing whole-prompt LRP attribution: aiming for {trials_per_prompt} successes per prompt.")

    for prompt_idx, generator in enumerate(trial_generators):
        successful_samples = 0
        pbar = tqdm(total=trials_per_prompt, desc=f"Whole-Prompt LRP Prompt {prompt_idx + 1}", leave=False)

        for prompt_data in generator:
            if successful_samples >= trials_per_prompt:
                break

            prompt_string = utils.create_prompt(prompt_data)
            target = prompt_data["query_target"]["output"]
            target = target[0] if isinstance(target, list) else target

            try:
                target_token_id = utils.get_answer_id(prompt_string, target, tokenizer)[0]
            except (IndexError, AssertionError):
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

                token_rank = utils.compute_individual_token_rank(next_token_logits, target_token_id)
                if token_rank > top_k_filter:
                    continue

                for attn_layer in outputs.attentions:
                    attn_layer.retain_grad()

                model.zero_grad(set_to_none=True)
                next_token_logits[target_token_id].backward()

                layer_scores = [
                    (a * a.grad)[0, :, -1, :].clamp(min=0).sum(dim=-1).float().detach()
                    for a in outputs.attentions
                ]
                all_relevances.append(torch.stack(layer_scores).cpu())

            successful_samples += 1
            pbar.update(1)

        pbar.close()

    if not all_relevances:
        raise ValueError("No valid combinations found where the target was in the Top-K predictions.")

    stacked_relevances = torch.stack(all_relevances)
    logger.info(f"Whole-prompt LRP finished. Collected {len(stacked_relevances)} total valid trials.")
    return stacked_relevances
