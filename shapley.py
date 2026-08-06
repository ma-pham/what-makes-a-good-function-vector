# shapley.py — one function replaces both of the collaborator's
from collections import defaultdict
import torch
from loguru import logger
from tqdm import tqdm
import utils


def _replace_heads_at_last_token(head_replacements, head_dim):
    """forward_pre_hook: overwrite several heads' slices of the o_proj input at the last token."""
    def pre_hook(_module, args):
        x = args[0].clone()
        for head, replacement in head_replacements:
            lo, hi = head * head_dim, (head + 1) * head_dim
            x[:, -1, lo:hi] = replacement.to(dtype=x.dtype, device=x.device)
        return (x,) + args[1:]
    return pre_hook


def shapley_values(model, tokenizer, model_config, trials, mean_acts, rng,
                   candidate_heads=None, coalition_size=4):
    """Approximated Shapley over `trials` (corrupt prompt_data — same list CIE uses).
    One random coalition per trial; marginal gain distributed to members.
    `rng` (numpy Generator) samples the coalitions. Returns (1, n_layers, n_heads)."""
    n_layers = model_config["n_layers"]
    n_heads = model_config["n_heads"]
    head_dim = model_config["head_dim"]
    layer_names = model_config["attn_hook_names"]
    device = model.device

    if candidate_heads is None:
        candidate_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    if len(candidate_heads) < coalition_size:
        raise ValueError("candidate pool must be >= coalition_size")

    sums = torch.zeros(n_layers, n_heads)
    counts = torch.zeros(n_layers, n_heads)
    logger.info(f"Computing Shapley over {len(trials)} permutations "
                f"(coalition {coalition_size}, pool {len(candidate_heads)})")

    for prompt_data in tqdm(trials, total=len(trials)):
        prompt_string = utils.create_prompt(prompt_data)
        target = prompt_data["query_target"]["output"]  # space-prepended -> clean token boundary
        target = target[0] if isinstance(target, list) else target
        target_token_id = utils.get_answer_id(prompt_string, target, tokenizer)[0]

        inputs = tokenizer(prompt_string, return_tensors="pt").to(device)
        clean_p = torch.softmax(model(**inputs).logits[:, -1, :], dim=-1)[0, target_token_id]

        group = [candidate_heads[i] for i in rng.choice(len(candidate_heads), coalition_size, replace=False)]
        by_layer = defaultdict(list)
        for l, h in group:
            by_layer[l].append((h, mean_acts[l, h]))

        handles = [model.get_submodule(layer_names[layer]).register_forward_pre_hook(
                       _replace_heads_at_last_token(reps, head_dim))
                   for layer, reps in by_layer.items()]
        try:
            logits = model(**inputs).logits[:, -1, :]
        finally:
            for h in handles:
                h.remove()

        gain = (torch.softmax(logits, dim=-1)[0, target_token_id] - clean_p).item()
        for l, h in group:
            sums[l, h] += gain
            counts[l, h] += 1

    return (sums / counts.clamp(min=1)).unsqueeze(0)  # (1, n_layers, n_heads)
