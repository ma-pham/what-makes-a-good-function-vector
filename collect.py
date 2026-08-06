"""Mean-activation collection, branch-agnostic.

Takes a list of `trials` (each a prompt_data dict from a source's clean_trials)
and averages a per-head o_proj-input activation over them.
Result shape in both cases: (n_layers, n_heads, head_dim).

Two collection sites:
  * mean_activations()    -- the last token, summed over ALL keys (the default).
  * mean_if_activations() -- the last token, restricted to the Intentional Frame keys.

They are the same vector under a different key range: the o_proj input of head h at
the last position is sum_j alpha[h,-1,j] * v[h,j], and the IF variant restricts that
sum to j in the IF token positions -- the same positions the IF attribution methods
score (lrp/attn_diff/frame_cie/if_patching all index [.., -1, key_idxs]).
"""

import torch
from loguru import logger

import utils
# the IF token lookup, shared with the attribution methods rather than copied again
from if_patching import get_substring_indices


def mean_activations(model, tokenizer, model_config, trials, batch_size=1):
    """Average the last-token per-head activation over `trials`."""
    logger.info(f"Collecting last-token activations over {len(trials)} trials (batch_size={batch_size})")

    storage = torch.zeros(
        len(trials),
        model_config["n_layers"],
        model_config["n_heads"],
        model_config["head_dim"],
    )

    tokenizer.padding_side = "left"  # last column = last real token for every row
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    layers = model_config["attn_hook_names"]

    for start in range(0, len(trials), batch_size):
        batch = trials[start : start + batch_size]

        sentences = []
        for prompt_data in batch:
            q = prompt_data["query_target"]["input"]
            q = q[0] if isinstance(q, list) else q
            sentences.append(utils.create_prompt(prompt_data, sentence=q))

        cache = {}

        def make_hook(name):
            def hook(_module, inp, _out):
                cache[name] = inp[0].detach()  # o_proj input: (batch, seq, n_heads*head_dim)
            return hook

        handles = [model.get_submodule(name).register_forward_hook(make_hook(name)) for name in layers]
        try:
            inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)
            model(**inputs)
            per_layer = [utils.split_heads(cache[name][:, -1, :], model_config["n_heads"]) for name in layers]
            stacked = torch.stack(per_layer, dim=1)  # (batch, n_layers, n_heads, head_dim)
        finally:
            for h in handles:
                h.remove()

        storage[start : start + len(batch)] = stacked.cpu()

    return storage.mean(dim=0)  # (n_layers, n_heads, head_dim)


@torch.no_grad()
def mean_if_activations(model, tokenizer, model_config, trials, search_strings,
                        min_hit_rate=0.5):
    """Average the IF-sourced part of the last-token per-head activation over `trials`.

    For head h the last token's o_proj input is sum_j alpha[h,-1,j] * v[h,j]; this keeps
    only the terms whose key j falls inside an occurrence of `search_strings`. Same shape
    and same space as mean_activations(), so it is a drop-in for eval.evaluate().

    Trials whose prompt contains no IF occurrence are skipped -- a mean over a handful of
    survivors is still a well-shaped tensor, so `min_hit_rate` makes that failure loud
    instead of silent. Requires attn_implementation="eager" (output_attentions).

    Returns (mean_acts, stats).
    """
    if not search_strings:
        raise ValueError("mean_if_activations needs a non-empty search_strings list")

    n_layers, n_heads = model_config["n_layers"], model_config["n_heads"]
    logger.info(f"Collecting IF-sourced last-token activations over {len(trials)} trials "
                f"(search_strings={list(search_strings)})")

    total = torch.zeros(n_layers, n_heads, model_config["head_dim"])
    hits, if_token_counts = 0, []

    for prompt_data in trials:
        q = prompt_data["query_target"]["input"]
        q = q[0] if isinstance(q, list) else q
        sentence = utils.create_prompt(prompt_data, sentence=q)

        key_idxs = get_substring_indices(sentence, tokenizer, search_strings)
        if not key_idxs:
            continue

        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
        cache = {}

        def make_hook(idx):
            def hook(_module, _inp, out):
                cache[idx] = out.detach()  # (batch, seq, n_kv_heads*head_dim)
            return hook

        handles = [model.get_submodule(name.replace("o_proj", "v_proj"))
                   .register_forward_hook(make_hook(i))
                   for i, name in enumerate(model_config["attn_hook_names"])]
        try:
            out = model(**inputs, output_attentions=True, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        for layer in range(n_layers):
            # attention of the last query position onto the IF keys: (n_heads, n_if)
            alpha = out.attentions[layer][0, :, -1, key_idxs]
            # v_proj output carries no RoPE, so it is exactly the V attention uses.
            # (seq, n_kv_heads, head_dim) -> (n_kv_heads, seq, head_dim); GQA shares
            # each kv head across a group of query heads
            v = cache[layer][0]  # (seq, n_kv_heads*head_dim)
            n_kv = v.shape[-1] // model_config["head_dim"]
            v = v.view(-1, n_kv, model_config["head_dim"]).permute(1, 0, 2)
            v = v.repeat_interleave(n_heads // n_kv, dim=0)[:, key_idxs, :]
            # sum_j alpha * v over the IF keys only -> (n_heads, head_dim)
            total[layer] += (alpha.unsqueeze(-1) * v).sum(dim=1).float().cpu()

        hits += 1
        if_token_counts.append(len(key_idxs))

    hit_rate = hits / max(len(trials), 1)
    if hit_rate < min_hit_rate:
        raise ValueError(f"IF found in only {hits}/{len(trials)} trials (hit rate "
                         f"{hit_rate:.0%} < {min_hit_rate:.0%}); the instructions selected here "
                         f"probably do not contain {list(search_strings)}")

    mean_if_tokens = sum(if_token_counts) / max(len(if_token_counts), 1)
    logger.info(f"IF hit {hits}/{len(trials)} trials ({hit_rate:.0%}), "
                f"mean {mean_if_tokens:.1f} IF tokens per prompt")
    return total / max(hits, 1), {"trials": len(trials), "hits": hits,
                                  "hit_rate": hit_rate, "mean_if_tokens": mean_if_tokens}