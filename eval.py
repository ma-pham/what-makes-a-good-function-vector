"""Zero-shot steering evaluation of the extracted task representation.

Two steering methods (pick ONE at a time via `steering_method`):

  * "dist" -- additive per-head steering: add the selected heads' clean mean
    activation back into their own o_proj input at the last token.
  * "fv"   -- function vector: sum o_proj(mean head act) across the selected
    heads into a single residual-stream vector, add it at one edit layer.

Both steer a *zero-shot* query ("Q: <input>\\nA:") and measure whether the model
now predicts the correct answer -- top-k accuracy over the test split.

The head set is chosen from `head_scores` (CIE or LRP; shape
(n_trials, n_layers, n_heads), reduced over trials here), and the steering
content from `mean_acts` (n_layers, n_heads, head_dim).

Can be used two ways:
  * from newmain, call `evaluate(...)` with the in-memory tensors, or
  * standalone: `python neweval.py` -> `main()` loads the cached
    {dataset}_{prompt_type}_mean_head_activations.pt and
    {dataset}_{prompt_type}_{method}_head_scores.pt and
    re-runs the eval (no recomputation).
"""

import os
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

import utils
from utils import set_seed  # was: from recipe.function_vectors.utils.model_utils import set_seed
import instruction
import demonstrator


# ---------------------------------------------------------------------------
# Head selection + steering-vector construction
# ---------------------------------------------------------------------------

def select_top_heads(head_scores, topk):
    """(n_trials, n_layers, n_heads) or (n_layers, n_heads) -> list of (layer, head)."""
    scores = head_scores
    while scores.dim() > 2:  # reduce any leading trial dimension(s)
        scores = scores.mean(dim=0)
    n_heads = scores.shape[1]
    idx = torch.topk(scores.flatten(), k=topk).indices
    return [(int(i) // n_heads, int(i) % n_heads) for i in idx]


def compute_fv(mean_acts, top_heads, model, model_config):
    """Sum o_proj(zero-padded mean head act) across top heads -> (resid_dim,).

    For each (L, H): place mean_acts[L, H] in head H's slice of a zero vector,
    push it through layer L's o_proj, and accumulate. Matches the reference's
    compute_function_vector.
    """
    resid_dim = model_config["resid_dim"]
    head_dim = model_config["head_dim"]
    n_heads = model_config["n_heads"]
    embed_dim = n_heads * head_dim
    device = model.device
    dtype = next(model.parameters()).dtype

    fv = torch.zeros(1, 1, resid_dim, device=device, dtype=dtype)
    for L, H in top_heads:
        o_proj = model.get_submodule(model_config["attn_hook_names"][L])
        x = torch.zeros(1, 1, embed_dim, device=device, dtype=dtype)
        x[0, 0, H * head_dim : (H + 1) * head_dim] = mean_acts[L, H].to(device=device, dtype=dtype)
        fv = fv + o_proj(x)
    return fv.reshape(resid_dim)


# ---------------------------------------------------------------------------
# Steered evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def eval_steered(model, tokenizer, model_config, dataset, steering_method, steering,
                 top_heads, edit_layer, scale, prefixes, separators, prepend_bos,
                 filter_set=None, max_examples=None, split="test"):
    """Run one steered pass over the eval split; return top-k accuracy + ranks."""
    n_heads = model_config["n_heads"]
    head_dim = model_config["head_dim"]
    device = model.device

    hooks = []
    if steering_method == "dist":
        # steering: (n_top_heads, head_dim) aligned with top_heads order
        heads_by_layer = {}
        for gid, (L, H) in enumerate(top_heads):
            heads_by_layer.setdefault(L, []).append((gid, H))
        for L, head_list in heads_by_layer.items():
            def attn_pre_hook(_module, inputs, _head_list=head_list):
                x = inputs[0]  # (B, S, n_heads*head_dim) = o_proj input
                B, S, D = x.shape
                xh = x.view(B, S, n_heads, head_dim)
                for gid, H in _head_list:
                    xh[:, -1, H] = xh[:, -1, H] + scale * steering[gid].to(x.device, x.dtype)
                return (xh.view(B, S, D),)
            o_proj = model.get_submodule(model_config["attn_hook_names"][L])
            hooks.append(o_proj.register_forward_pre_hook(attn_pre_hook))
    elif steering_method == "fv":
        fv_scaled = (scale * steering).to(device)

        def fv_hook(_module, _inp, out):
            # decoder layer may return a tensor or a (hidden, ...) tuple
            if isinstance(out, tuple):
                hidden = out[0].clone()
                hidden[:, -1, :] = hidden[:, -1, :] + fv_scaled
                return (hidden,) + out[1:]
            out = out.clone()
            out[:, -1, :] = out[:, -1, :] + fv_scaled
            return out
        layer = model.get_submodule(model_config["layer_hook_names"][edit_layer])
        hooks.append(layer.register_forward_hook(fv_hook))
    else:
        raise ValueError(f"steering_method must be 'dist' or 'fv', got {steering_method!r}")

    split_data = dataset[split]
    indices = list(filter_set) if filter_set is not None else list(range(len(split_data)))
    if max_examples is not None:
        indices = indices[:max_examples]

    ranks = []
    try:
        for j in tqdm(indices, desc=f"{steering_method} scale={scale}", leave=False):
            sample = split_data[int(j)]
            prompt_data = utils.word_pairs_to_prompt_data(
                {"input": [], "output": []}, query_target_pair=sample, instructions="",
                prepend_bos_token=prepend_bos, shuffle_labels=False,
                prefixes=prefixes, separators=separators, tokenizer=tokenizer,
            )
            target = prompt_data["query_target"]["output"]
            target = target[0] if isinstance(target, list) else target
            sentence = utils.create_prompt(prompt_data)
            try:
                target_token_id = utils.get_answer_id(sentence, target, tokenizer)[0]
            except (IndexError, AssertionError):
                continue

            toks = tokenizer(sentence, return_tensors="pt").to(device)
            logits = model(input_ids=toks["input_ids"]).logits[:, -1, :].float()
            ranks.append(utils.compute_individual_token_rank(logits, target_token_id))
    finally:
        for h in hooks:
            h.remove()

    return {
        "topk": {K: float(utils.compute_top_k_accuracy(ranks, K)) for K in (1, 2, 3)},
        "ranks": ranks,
        "n_examples": len(ranks),
    }


def evaluate(model, tokenizer, model_config, dataset, mean_acts, head_scores,
             steering_method, topk_heads, edit_layer, scales,
             prefixes, separators, prepend_bos, filter_set=None, max_examples=None):
    """Select heads, build the steering signal for `steering_method`, and sweep scales.

    Returns (results_by_scale, top_heads).
    """
    top_heads = select_top_heads(head_scores, topk_heads)
    logger.info(f"Top {len(top_heads)} heads (layer, head): {top_heads}")

    if steering_method == "dist":
        n_heads, head_dim = model_config["n_heads"], model_config["head_dim"]
        flat = mean_acts.reshape(-1, head_dim)  # (n_layers*n_heads, head_dim)
        steering = flat[[L * n_heads + H for L, H in top_heads]]  # (n_top_heads, head_dim)
    elif steering_method == "fv":
        steering = compute_fv(mean_acts, top_heads, model, model_config)
    else:
        raise ValueError(f"steering_method must be 'dist' or 'fv', got {steering_method!r}")

    results = {}
    for scale in scales:
        r = eval_steered(
            model, tokenizer, model_config, dataset, steering_method, steering,
            top_heads, edit_layer, scale, prefixes, separators, prepend_bos,
            filter_set=filter_set, max_examples=max_examples,
        )
        results[scale] = r
        t = r["topk"]
        logger.info(f"[{steering_method} scale={scale}] top1={t[1]:.3f} top2={t[2]:.3f} "
                    f"top3={t[3]:.3f}  (n={r['n_examples']})")
    return results, top_heads


# ---------------------------------------------------------------------------
# Standalone entry point: load cached artifacts and re-run the eval
# ---------------------------------------------------------------------------

def main():
    # --- hardcoded config (mirror newmain) -----------------------------------
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    dataset_name = "antonym"
    prompt_length = "short"
    device = "cuda:0"
    seed = 42
    test_split = 0.3
    STORAGE_ROOT = "storage"

    prompt_type = "demonstrator"  # which branch's artifacts to load: "demonstrator" or "instruction"
    attribution_method = "cie"  # which head_scores file to load: "cie" or "lrp"
    steering_method = "fv"      # "fv" or "dist"
    topk_heads = 10
    fv_edit_layer = 9
    eval_scales = [1.0]
    eval_max_examples = None
    # -------------------------------------------------------------------------

    short_model_name = model_name[model_name.rfind("/") + 1 :]
    root_data_dir = f"{STORAGE_ROOT}/function_vectors/dataset_files"
    save_dir = utils.result_dir(STORAGE_ROOT, short_model_name, dataset_name, prompt_length)

    prefixes = {"input": "Q:", "output": "A:", "instructions": ""}
    separators = {"input": "\n", "output": "\n\n", "instructions": "\n"}

    torch.set_grad_enabled(False)
    set_seed(seed)

    logger.info("Loading model")
    model, tokenizer, model_config = utils.load_gpt_model_and_tokenizer(model_name, device=device)
    prepend_bos = not model_config["prepend_bos"]

    logger.info("Loading dataset")
    dataset = utils.load_dataset(
        # split pinned to utils.SPLIT_SEED; `seed` drives the stochastic stages only
        dataset_name, root_data_dir=root_data_dir, test_size=test_split, split_valid=True
    )

    # load cached artifacts (no recomputation); prompt_type in the name so demonstrator
    # and instruction artifacts don't collide (must match newmain's paths)
    mact_path = Path(save_dir) / f"{dataset_name}_{prompt_type}_mean_head_activations.pt"
    scores_path = Path(save_dir) / f"{dataset_name}_{prompt_type}_{attribution_method}_head_scores.pt"
    logger.info(f"Loading mean activations: {mact_path}")
    mean_acts = torch.load(mact_path, map_location=device)
    logger.info(f"Loading head scores: {scores_path}")
    head_scores = torch.load(scores_path, map_location="cpu")

    # test filter via the branch's (self-validating) test_filter_set rather than reading
    # the raw JSON -- reuses the cache when the selection matches, recomputes otherwise
    n_best_instructions = 5
    n_icl_examples = 10
    if prompt_type == "instruction":
        candidate_instructions = instruction.load_candidate_prompts(dataset_name, STORAGE_ROOT)
        instructions, _ = instruction.select(
            model, tokenizer, dataset, candidate_instructions, prefixes=prefixes, separators=separators,
            n_best_instructions=n_best_instructions, prepend_bos=prepend_bos, query_split="train",
            save_dir=save_dir, dataset_name=dataset_name)
        test_fs = instruction.test_filter_set(
            model, tokenizer, dataset, instructions, prefixes=prefixes, separators=separators,
            prepend_bos=prepend_bos, save_dir=save_dir, dataset_name=dataset_name).tolist()
    else:  # demonstrator
        test_fs = demonstrator.test_filter_set(
            model, tokenizer, dataset, prefixes=prefixes, separators=separators,
            n_icl_examples=n_icl_examples, prepend_bos=prepend_bos,
            save_dir=save_dir, dataset_name=dataset_name).tolist()
    logger.info(f"Test filter set: {len(test_fs)} examples")

    results, top_heads = evaluate(
        model, tokenizer, model_config, dataset, mean_acts, head_scores,
        steering_method=steering_method, topk_heads=topk_heads,
        edit_layer=fv_edit_layer, scales=eval_scales,
        prefixes=prefixes, separators=separators, prepend_bos=prepend_bos,
        filter_set=test_fs, max_examples=eval_max_examples,
    )

    out_path = Path(save_dir) / f"{dataset_name}_{prompt_type}_{attribution_method}_{steering_method}_eval.pt"
    torch.save({"results": results, "top_heads": top_heads}, out_path)
    logger.info(f"Saved eval results -> {out_path}")
    return results


if __name__ == "__main__":
    main()
