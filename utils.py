import numpy as np
import hashlib
import json
from datetime import datetime
from functools import lru_cache
import random
import re
import torch
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from loguru import logger
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Dict
from torch.nn import functional as F


def load_json(path):
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=None)
def load_json_or_empty(path: str) -> dict:
    """Like load_json, but a missing file warns and yields {} instead of raising.
    Cached, so repeated lookups of the same config file cost one read. Treat the
    result as read-only -- every caller shares it."""
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    logger.warning(f"JSON not found at {p}; using an empty dict")
    return {}


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def min_filter_examples(total_activation_examples, n_cie_trials, n_best_instructions) -> int:
    """Smallest filter set the trial budget can spend on distinct examples: below this,
    per-instruction draws repeat queries and the stats are pseudo-replicated. Mirrors the
    original pipeline's min_passing_examples."""
    return max(total_activation_examples // n_best_instructions,
               n_cie_trials // n_best_instructions)


def write_failure_marker(save_dir, code, reason):
    """Marker file matching the original pipeline's schema/filename, so this storage
    tree is machine-readable the same way the remote one is."""
    save_json(Path(save_dir) / "equiprobable_failure.json", {
        "code": code,
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
    })


def load_cached_tensor(path, expected_ndim=None, force=False):
    """Cached tensor at `path`, or None if missing / force=True / rank != expected_ndim
    (e.g. an original-pipeline 4D mean_head_activations.pt where we expect 3D).
    Pair with save_tensor(): `t = load(...)` / `if t is None: t = save(p, compute())`."""
    path = Path(path)
    if force or not path.exists():
        return None
    t = torch.load(path, map_location="cpu")
    if expected_ndim is not None and t.dim() != expected_ndim:
        logger.warning(f"cached tensor {path} has ndim {t.dim()} (expected {expected_ndim}); recomputing")
        return None
    logger.info(f"loaded cached tensor from {path} {tuple(t.shape)}")
    return t


def save_tensor(path, t):
    """Persist `t` to `path` (creating parents) and return it, so it can wrap a compute
    expression inline: `x = save_tensor(p, compute())`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t, path)
    logger.info(f"saved tensor -> {path} {tuple(t.shape)}")
    return t


def load_or_compute_tensor(path, compute_fn, expected_ndim=None, force=False):
    """Reuse a cached tensor at `path`, else compute via `compute_fn()` and save it.

    If `expected_ndim` is given and a cached tensor has a different rank (e.g. an
    original-pipeline 4D mean_head_activations.pt where we expect 3D), it is ignored
    and recomputed. `force=True` always recomputes.
    """
    path = Path(path)
    if path.exists() and not force:
        t = torch.load(path, map_location="cpu")
        if expected_ndim is not None and t.dim() != expected_ndim:
            logger.warning(f"cached tensor {path} has ndim {t.dim()} (expected {expected_ndim}); recomputing")
        else:
            logger.info(f"loaded cached tensor from {path} {tuple(t.shape)}")
            return t
    t = compute_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t, path)
    logger.info(f"saved tensor -> {path} {tuple(t.shape)}")
    return t


def set_seed(seed: int) -> None:
    """Global determinism baseline. Two jobs:

    1. cuDNN determinism flags (deterministic kernels for the forward pass) -- the
       part that actually reduces run-to-run forward noise; bf16 GPU reductions
       remain an irreducible floor.
    2. seed the global np/random/torch streams as insurance for any draw not routed
       through an explicit make_rng() generator.

    The stochastic *sampling* is made reproducible by the per-stage generators, not by
    this -- so results no longer depend on RNG-consumption order or cache state.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # must be False for determinism (was True)


def make_rng(seed: int, *keys) -> np.random.Generator:
    """Generator seeded from (seed, *keys) -- e.g. make_rng(42, task, "cie"). Gives each
    stage its own stream, so its draws don't depend on what other stages consumed or
    cache-hit. String keys use sha1, not Python's per-process-salted hash()."""
    mixed = [seed]
    for k in keys:
        if isinstance(k, str):
            k = int.from_bytes(hashlib.sha1(k.encode()).digest()[:8], "big")
        mixed.append(int(k))
    return np.random.default_rng(np.random.SeedSequence(mixed))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _resolve_dtype(config) -> torch.dtype:
    """Always bfloat16, for cross-model comparability; warns if the checkpoint differs."""
    declared = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if isinstance(declared, str):
        declared = getattr(torch, declared, None)
    if isinstance(declared, torch.dtype) and declared != torch.bfloat16:
        logger.warning(f"checkpoint declares {declared}; loading bfloat16 anyway for comparability")
    elif not isinstance(declared, torch.dtype):
        logger.warning("checkpoint declares no dtype; using bfloat16")
    return torch.bfloat16


def _tokenizer_handles_bos(tokenizer) -> bool:
    """True if BOS is already handled -- tokenizer adds it (Llama) or there is none
    (Qwen3). Callers invert this (`not model_config["prepend_bos"]`) to decide whether
    to prepend one themselves; doing so when bos_token is None would raise TypeError."""
    if tokenizer.bos_token_id is None:
        return True
    ids = tokenizer("probe").input_ids
    return bool(ids) and ids[0] == tokenizer.bos_token_id


def load_gpt_model_and_tokenizer(model_name: str, device="cuda", attn_implementation=None):
    """Load a HuggingFace causal LM + tokenizer and a standardized MODEL_CONFIG.

    Pass attn_implementation="eager" for LRP attribution (see newlrp.py): only the
    eager attention path returns gradient-carrying attention matrices.
    """
    assert model_name is not None
    logger.info(f"Loading: {model_name}")

    lname = model_name.lower()
    # the family check guards the hook-name assumptions below (model.layers.N.self_attn.o_proj),
    # not the dtype -- dtype and BOS handling are derived from the model/tokenizer instead
    if not any(fam in lname for fam in ("llama", "olmo", "qwen", "gemma")):
        raise NotImplementedError(f"Model not wired up yet: {model_name}")

    model_dtype = _resolve_dtype(AutoConfig.from_pretrained(model_name))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # needed only for batched activation collection (batch_size > 1)
        tokenizer.pad_token = tokenizer.eos_token
    prepend_bos = _tokenizer_handles_bos(tokenizer)

    logger.info(f"dtype={model_dtype} | tokenizer handles BOS: {prepend_bos}")
    # transformers renamed this kwarg: `torch_dtype` up to 4.x, `dtype` from 5.x.
    # lxt (the LRP branch) only supports 4.x, so both have to load.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=model_dtype, attn_implementation=attn_implementation
        ).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=model_dtype, attn_implementation=attn_implementation
        ).to(device)

    n_layers = model.config.num_hidden_layers
    MODEL_CONFIG = {
        "n_heads": model.config.num_attention_heads,
        "n_layers": n_layers,
        "resid_dim": model.config.hidden_size,
        "name_or_path": model.config._name_or_path,
        "attn_hook_names": [f"model.layers.{layer}.self_attn.o_proj" for layer in range(n_layers)],
        "layer_hook_names": [f"model.layers.{layer}" for layer in range(n_layers)],
        "prepend_bos": prepend_bos,
        "head_dim": getattr(
            model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads
        ),
    }
    return model, tokenizer, MODEL_CONFIG


# ---------------------------------------------------------------------------
# Dataset utils (unchanged from distfvnew)
# ---------------------------------------------------------------------------
class ICLDataset:
    """A simple dataset of input->output pairs, used for ICL prompt construction."""

    def __init__(self, dataset):
        if isinstance(dataset, str):
            self.raw_data = pd.read_json(dataset)
        elif isinstance(dataset, dict):
            self.raw_data = pd.DataFrame(dataset)
        elif isinstance(dataset, pd.DataFrame):
            self.raw_data = dataset
        else:
            raise ValueError(f"Unrecognized ICLDataset type: {type(dataset)}")
        self.raw_data = self.raw_data[["input", "output"]]

    def subset(self, start=0, end=None):
        if end is None:
            end = len(self.raw_data)
        return ICLDataset(self.raw_data.iloc[start:end].copy(deep=True).reset_index(drop=True))

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.raw_data.iloc[i].to_dict()
        elif isinstance(i, slice):
            return self.raw_data.iloc[i].to_dict(orient="list")
        elif isinstance(i, (list, np.ndarray)):
            return self.raw_data.iloc[i].to_dict(orient="list")
        elif isinstance(i, str):
            if i not in self.raw_data.columns:
                raise KeyError(
                    f"Column '{i}' not in the dataset. Columns: {self.raw_data.columns.to_list()}"
                )
            return self.raw_data[i].to_list()
        raise ValueError(f"{i} is not a valid index type. Expected int/list/np.ndarray/slice/str")

    def __len__(self):
        return len(self.raw_data)

    def __repr__(self):
        return (
            "ICLDataset({\n\tfeatures: "
            + f"{self.raw_data.columns.to_list()},\n\tnum_rows: {len(self)}\n}})"
        )


def split_icl_dataset(
    dataset, train_size=None, test_size=0.3, valid_from_train_size=None, seed=42, split_valid=True
) -> Dict[str, "ICLDataset"]:
    if train_size is None and test_size is None:
        train_size, test_size = 0.7, 0.3
    elif train_size is not None and test_size is None:
        test_size = 1 - train_size
    elif train_size is None and test_size is not None:
        train_size = 1 - test_size
    else:
        assert train_size + test_size == 1

    if valid_from_train_size is None:
        valid_from_train_size = (test_size**2) / (1 - test_size)
    else:
        assert 0 < valid_from_train_size < 1

    train, test = train_test_split(dataset.raw_data, test_size=test_size, random_state=seed)
    if split_valid:
        train, valid = train_test_split(train, test_size=valid_from_train_size, random_state=seed)
        valid = ICLDataset(valid.to_dict(orient="list"))
    train = ICLDataset(train.to_dict(orient="list"))
    test = ICLDataset(test.to_dict(orient="list"))
    return {"train": train, "valid": valid, "test": test} if split_valid else {"train": train, "test": test}


def find_dataset_folder(task_name: str, root_data_dir: str = "../dataset_files"):
    data_folders = ["abstractive", "extractive"]
    d_group = [
        dt for dt in data_folders
        if os.path.exists(os.path.join(root_data_dir, dt, task_name + ".json"))
    ]
    return d_group[0] if len(d_group) == 1 else None


SPLIT_SEED = 42
"""Fixes train/valid/test membership, and must stay constant across runs.

The cached artifacts -- per-prompt ranks, filter sets, mean activations -- are all
*positional* into the split, and none of them record the seed they were built under.
Changing this therefore does not just reshuffle the data, it silently re-points every
cached index at a different example. Vary the run seed instead (trial sampling, CIE
draws, shapley permutations): that isolates head-selection stability from dataset
variance, which is what a seed sweep is supposed to measure.
"""


def load_dataset(
    task_name: str, root_data_dir: str = "../dataset_files", test_size=0.3, seed=SPLIT_SEED, split_valid=True
) -> Dict[str, "ICLDataset"]:
    assert test_size <= 1.0
    dataset_folder = find_dataset_folder(task_name, root_data_dir)
    assert dataset_folder is not None, (
        f"'task_name'={task_name}.json must be uniquely in one of ['abstractive','extractive'] under {root_data_dir}"
    )
    d_path = os.path.join(Path(root_data_dir), dataset_folder, f"{task_name}.json")
    dataset = ICLDataset(d_path)
    return split_icl_dataset(dataset, test_size=test_size, seed=seed, split_valid=split_valid)


# ---------------------------------------------------------------------------



def result_dir(storage_root, model_short_name, dataset_name, prompt_length="short"):
    """Directory mirroring the original: full_results_prompt_based_{length}/{model}/{dataset}."""
    return (
        Path(storage_root)
        / "function_vectors"
        / f"full_results_prompt_based_{prompt_length}"
        / model_short_name
        / dataset_name
    )


def word_pairs_to_prompt_data(
    word_pairs: dict,
    instructions: str = "",
    prefixes: dict = {"input": "Q:", "output": "A:", "instructions": ""},
    separators: dict = {"input": "\n", "output": "\n\n", "instructions": ""},
    query_target_pair: dict = None,
    prepend_bos_token=False,
    shuffle_labels=False,
    prepend_space=True,
    tokenizer_bos_token: str | None = None,
    tokenizer=None,
) -> dict:
    """Takes a dataset of word pairs, and constructs a prompt_data dict with additional information to construct an ICL prompt.
    Parameters:
    word_pairs: dict of the form {'word1':['a', 'b', ...], 'word2':['c', 'd', ...]}
    instructions: prefix instructions for an ICL prompt
    prefixes: dict of ICL prefixes that are prepended to inputs, outputs and instructions
    separators: dict of ICL separators that are appended to inputs, outputs and instructions
    query_target_pair: dict with a single input-output pair acting as the query for the prompt
    prepend_bos_token: whether or not to prepend a BOS token to the prompt
    shuffle_labels: whether to shuffle the ICL labels
    prepend_space: whether to prepend a space to every input and output token

    Returns:
    prompt_data: dict containing ICL prompt examples, and template information
    """
    if tokenizer_bos_token is None:
        if tokenizer is not None:
            tokenizer_bos_token = tokenizer.bos_token
        elif prepend_bos_token:
            raise ValueError("'prepend_bos_token' is set to True, but no tokenizer or bos_token was provided.")

    prompt_data = {}
    prompt_data["instructions"] = instructions
    prompt_data["separators"] = separators
    if prepend_bos_token:
        prefixes = {k: (v if k != "instructions" else tokenizer_bos_token + v) for (k, v) in prefixes.items()}
    prompt_data["prefixes"] = prefixes

    if query_target_pair is not None:
        query_target_pair = {k: (v[0] if isinstance(v, list) else v) for k, v in query_target_pair.items()}
    prompt_data["query_target"] = query_target_pair

    pairs = word_pairs.values()
    if shuffle_labels:
        pairs = [
            np.random.permutation(x).tolist() if i == 1 else x for (i, x) in enumerate(list(pairs))]  # shuffle labels only

    if prepend_space:
        prompt_data["examples"] = [{"input": " " + str(w1), "output": " " + str(w2)} for (w1, w2) in list(zip(*pairs))]
        prompt_data["query_target"] = (
            {k: " " + str(v) for k, v in query_target_pair.items()} if query_target_pair is not None else None
        )
    else:
        prompt_data["examples"] = [{"input": w1, "output": w2} for (w1, w2) in list(zip(*pairs))]

    return prompt_data

def create_fewshot_primer(prompt_data) -> str:
    """Creates the primer string for GPT in-context learning

    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information

    Returns:
    prompt: the constructed ICL prompt primer as a string
    """
    prompt = ""
    prompt += (
        prompt_data["prefixes"]["instructions"]
        + prompt_data["instructions"]
        + prompt_data["separators"]["instructions"]
    )

    for example in prompt_data["examples"]:
        prompt += prompt_data["prefixes"]["input"] + example["input"] + prompt_data["separators"]["input"]
        prompt += prompt_data["prefixes"]["output"] + example["output"] + prompt_data["separators"]["output"]

    return prompt


def create_prompt(prompt_data, sentence=None) -> str:
    """Creates a prompt using the specified sentence for GPT in-context learning

    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    sentence: a query string (sentence/word) to include in the ICL prompt

    Returns:
    prompt: the constructed ICL prompt as a string
    """
    if sentence is None and prompt_data["query_target"] is not None:
        sentence = prompt_data["query_target"]["input"]

    if isinstance(sentence, list):
        sentence = sentence[0]

    prompt_init = create_fewshot_primer(prompt_data)
    prompt = prompt_init + prompt_data["prefixes"]["input"] + sentence + prompt_data["separators"]["input"]
    prompt += prompt_data["prefixes"]["output"]

    return prompt


def compute_top_k_accuracy(target_token_ranks, k=10) -> float:
    """
    Evaluation to compute topk accuracy.

    Parameters:
    target_token_ranks: the distribution of output token ranks
    k: how many tokens we're looking at (top K)

    Return:
    The accuracy of the token in the top k of tokens
    """

    target_token_ranks = np.array(target_token_ranks)
    return (target_token_ranks < k).sum(axis=0) / len(target_token_ranks)


def compute_individual_token_rank(prob_dist, target_token_id) -> int:
    """
    Individual computation of token ranks across a single distribution.

    Parameters:
    prob_dist: the distribution of scores for a single output
    target_token_id: the target id we care about

    Return:
    A single value representing the token rank for that single token
    """
    if isinstance(target_token_id, list):
        target_token_id = target_token_id[0]

    return torch.where(  # type: ignore
        torch.argsort(prob_dist.squeeze(), descending=True) == target_token_id
    )[0].item()

def get_answer_id(query, answer, tokenizer):
    """
    Parameters:
    query (str): query as a string
    answer (str): expected answer as a string
    tokenizer: huggingface tokenizer

    Returns:
    target_token_ids (list): A list of the contextualized tokens of the answer
    """
    source = tokenizer(query, truncation=False, padding=False).input_ids
    target = tokenizer(query + answer, truncation=False, padding=False).input_ids
    assert len(source) < len(target) < tokenizer.model_max_length
    target_token_ids = target[len(source) :]
    return target_token_ids


def batch_target_token_ids(sentences, targets, tokenizer):
    """First target-token id per (sentence, target) pair -- batched get_answer_id(...)[0].
    Two batch tokenizer calls instead of two round-trips per example; same
    prefix-property assumption and assert as get_answer_id."""
    if not sentences:
        return []
    src = tokenizer(sentences, truncation=False, padding=False).input_ids
    full = tokenizer([s + t for s, t in zip(sentences, targets)],
                     truncation=False, padding=False).input_ids
    ids = []
    for s, f in zip(src, full):
        assert len(s) < len(f) < tokenizer.model_max_length
        ids.append(f[len(s)])
    return ids


@torch.inference_mode()
def batch_ranks(sentences, target_token_ids, model, tokenizer):
    """target-token rank per sentence: one batched forward, one comparison, one sync.
    rank = vocab entries scoring strictly higher than the target token -- same as the
    argsort rank except on exact ties, which this resolves in the target token's favour.
    Left-padding keeps the answer position at the last column for every row."""
    if not sentences:
        return []
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        enc = tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        ids = torch.tensor(target_token_ids, device=logits.device).unsqueeze(1)
        return (logits > logits.gather(1, ids)).sum(dim=1).tolist()
    finally:
        tokenizer.padding_side = prev_side

def should_skip_prompt(target: str, instruction: str):
    """True if `target` occurs as a standalone word in `instruction` (answer leakage).

    The target is stripped first so every caller gets the same verdict whether it
    passes the raw dataset value ("A") or the space-prepended prompt form (" A") --
    the decision-time masks in instruction.py must agree on this predicate.
    """
    t = str(target).strip().lower()
    if not t:
        return False
    p = instruction.lower()
    if re.search(f"[\\s'\"]({re.escape(t)})[\\s'\".,]", p) is not None:
        return True
    return p.startswith(t + " ")


def split_heads(activations, n_heads):
    """(batch, n_heads*head_dim) -> (batch, n_heads, head_dim)."""
    return activations.view(activations.shape[0], n_heads, activations.shape[-1] // n_heads)