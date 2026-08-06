"""Demonstrator (ICL) branch: select + test_filter_set + trial construction.

A trial is N input->output demos + a query. clean_trials (shuffle_labels=False) feed
collection/LRP; corrupt_trials (shuffle_labels=True) feed CIE.
"""

from pathlib import Path

import numpy as np
from loguru import logger

import utils


def _icl_ranks(model, tokenizer, dataset, split, prefixes, separators, n_shots, prepend_bos, batch_size):
    """Gold-token rank of every `split` example under a clean n-shot ICL prompt."""
    train_split = dataset["train"]
    query = dataset[split]

    sentences, targets = [], []
    for i in range(len(query)):
        demonstrators = (
            {"input": [], "output": []}
            if n_shots == 0
            else train_split[np.random.choice(len(train_split), n_shots, replace=False)]
        )
        prompt_data = utils.word_pairs_to_prompt_data(word_pairs=demonstrators, prefixes=prefixes, separators=separators, query_target_pair=query[i], prepend_bos_token=prepend_bos, shuffle_labels=False)
        target = prompt_data["query_target"]["output"]
        target = target[0] if isinstance(target, list) else target
        sentences.append(utils.create_prompt(prompt_data))
        targets.append(target)

    # gold-token ids in two batch tokenizer calls instead of two per example
    target_token_ids = utils.batch_target_token_ids(sentences, targets, tokenizer)

    ranks = []
    for start in range(0, len(sentences), batch_size):
        ranks.extend(utils.batch_ranks(sentences[start : start + batch_size],
                                       target_token_ids[start : start + batch_size], model, tokenizer))
    return ranks


def _filter_from_ranks(ranks, label):
    filter_set = np.atleast_1d(np.argwhere(np.asarray(ranks) == 0).squeeze()).astype(int)
    logger.info(f"[{label}] {len(filter_set)} usable examples")
    return filter_set


def select(model, tokenizer, dataset, prefixes, separators, n_icl_examples, prepend_bos,
           query_split="valid", n_shots=None, save_dir=None, dataset_name=None,
           batch_size=1, force=False):
    """Filter query split to ICL-solvable examples. Cached to {ds}_demonstrator_select.json.

    Reports size only; the minimum-size policy lives in the caller (see main.py).
    """
    n_shots = n_icl_examples if n_shots is None else n_shots
    cache_path = _cache_path(save_dir, dataset_name, "select")
    if cache_path is not None and cache_path.exists() and not force:
        logger.info(f"[select] demonstrator: loading cached filter set from {cache_path}")
        return np.asarray(utils.load_json(cache_path)["filter_set"], dtype=int)

    logger.info(f"[select] demonstrator: filtering '{query_split}' via {n_shots}-shot ICL")
    ranks = _icl_ranks(model, tokenizer, dataset, query_split, prefixes, separators, n_shots, prepend_bos, batch_size)
    filter_set = _filter_from_ranks(ranks, "select demonstrator")
    _save_filter(cache_path, filter_set, ranks, "select demonstrator")
    return filter_set


def test_filter_set(model, tokenizer, dataset, prefixes, separators, n_icl_examples, prepend_bos,
                    save_dir=None, dataset_name=None, batch_size=1, force=False, split="test"):
    """Eval filter: `split` examples solvable under clean n-shot ICL.
    Cached to {ds}_demonstrator_test_filter_set.json."""
    cache_path = _cache_path(save_dir, dataset_name, "test_filter_set")
    if cache_path is not None and cache_path.exists() and not force:
        logger.info(f"[test_filter] demonstrator: loading cached test filter set from {cache_path}")
        return np.asarray(utils.load_json(cache_path)["filter_set"], dtype=int)

    logger.info(f"[test_filter] demonstrator: filtering '{split}' via {n_icl_examples}-shot ICL")
    ranks = _icl_ranks(model, tokenizer, dataset, split, prefixes, separators, n_icl_examples, prepend_bos, batch_size)
    filter_set = _filter_from_ranks(ranks, "test_filter demonstrator")
    _save_filter(cache_path, filter_set, ranks, "test_filter demonstrator")
    return filter_set


def _save_filter(cache_path, filter_set, ranks, label):
    if cache_path is None:
        return
    utils.save_json(cache_path, {
        "filter_set": filter_set.tolist(),
        "clean_topk": [[k, utils.compute_top_k_accuracy(ranks, k)] for k in range(1, 4)],
    })
    logger.info(f"[{label}] cached -> {cache_path}")


def _cache_path(save_dir, dataset_name, kind):
    if save_dir is None or dataset_name is None:
        return None
    return Path(save_dir) / f"{dataset_name}_demonstrator_{kind}.json"


# --- trial construction (a trial == one prompt_data dict) ---

def _trial(dataset, n_icl_examples, prefixes, separators, prepend_bos, tokenizer,
           filter_set, query_split, shuffle_labels):
    word_pairs = dataset["train"][
        np.random.choice(len(dataset["train"]), n_icl_examples, replace=False)
    ]
    query = dataset[query_split][np.random.choice(filter_set, 1, replace=False)]
    return utils.word_pairs_to_prompt_data(
        word_pairs, query_target_pair=query, shuffle_labels=shuffle_labels,
        prepend_bos_token=prepend_bos, prefixes=prefixes, separators=separators, tokenizer=tokenizer)


def make_clean_trial(dataset, n_icl_examples, prefixes, separators, prepend_bos, tokenizer,
                     filter_set=None, query_split="valid"):
    """A single clean trial (shuffle_labels=False) -- for LRP's retry loop."""
    fs = filter_set if filter_set is not None else np.arange(len(dataset[query_split]))
    return _trial(dataset, n_icl_examples, prefixes, separators, prepend_bos, tokenizer, fs, query_split, False)


def clean_trials(dataset, n_trials, n_icl_examples, prefixes, separators, prepend_bos, tokenizer,
                 filter_set=None, query_split="valid"):
    """`n_trials` clean trials -- for mean-activation collection."""
    fs = filter_set if filter_set is not None else np.arange(len(dataset[query_split]))
    return [_trial(dataset, n_icl_examples, prefixes, separators, prepend_bos, tokenizer, fs, query_split, False)
            for _ in range(n_trials)]


def corrupt_trials(dataset, n_trials, n_icl_examples, prefixes, separators, prepend_bos, tokenizer,
                   filter_set=None, query_split="valid"):
    """`n_trials` corrupted trials (shuffle_labels=True) -- for CIE."""
    fs = filter_set if filter_set is not None else np.arange(len(dataset[query_split]))
    return [_trial(dataset, n_icl_examples, prefixes, separators, prepend_bos, tokenizer, fs, query_split, True)
            for _ in range(n_trials)]
