"""Causal indirect effect (CIE) of each attention head, branch-agnostic.

Takes a list of *corrupt* `trials` (label-shuffled demos, or empty/equiprobable
instruction) and, for every (layer, head), patches that head's clean mean
activation back in at the last token, recording ΔP(correct answer).
Returns (n_trials, n_layers, n_heads).
"""

import torch
from loguru import logger
from tqdm import tqdm

import utils


def _replace_head_at_last_token(head, head_dim, replacement):
    """forward_pre_hook: overwrite head `head`'s slice of the o_proj input at the
    last token with `replacement` (shape (head_dim,))."""
    lo, hi = head * head_dim, (head + 1) * head_dim

    def pre_hook(_module, args):
        x = args[0].clone()  # (batch, seq, n_heads*head_dim)
        x[:, -1, lo:hi] = replacement.to(dtype=x.dtype, device=x.device)
        return (x,) + args[1:]

    return pre_hook


def indirect_effect(model, tokenizer, model_config, trials, mean_acts):
    """CIE of every head over `trials` (corrupt prompt_data). mean_acts: (n_layers, n_heads, head_dim)."""
    n_layers = model_config["n_layers"]
    n_heads = model_config["n_heads"]
    head_dim = model_config["head_dim"]
    layer_names = model_config["attn_hook_names"]
    device = model.device

    indirect_effect = torch.zeros(len(trials), n_layers, n_heads)
    logger.info(f"Computing indirect effect over {len(trials)} trials ({n_layers}x{n_heads} heads each)")

    for t, prompt_data in enumerate(tqdm(trials, total=len(trials))):
        prompt_string = utils.create_prompt(prompt_data)
        target = prompt_data["query_target"]["output"]  # space-prepended -> clean token boundary
        target = target[0] if isinstance(target, list) else target
        target_token_id = utils.get_answer_id(prompt_string, target, tokenizer)[0]

        inputs = tokenizer(prompt_string, return_tensors="pt").to(device)
        clean_p = torch.softmax(model(**inputs).logits[:, -1, :], dim=-1)[0, target_token_id]

        for layer in range(n_layers):
            submodule = model.get_submodule(layer_names[layer])
            for head in range(n_heads):
                hook = _replace_head_at_last_token(head, head_dim, mean_acts[layer, head])
                handle = submodule.register_forward_pre_hook(hook)
                try:
                    logits = model(**inputs).logits[:, -1, :]
                finally:
                    handle.remove()
                interv_p = torch.softmax(logits, dim=-1)[0, target_token_id]
                indirect_effect[t, layer, head] = (interv_p - clean_p).item()

    return indirect_effect
