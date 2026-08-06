"""Per-prompt variant of extract_auto_if.py.

Same word-masking -> Intentional Frame extraction, but the per-prompt IFs are
NOT unioned into a single set per task. Each prompt keeps its own filter, so a
consumer can pair instruction i with the IF extracted from instruction i.
"""

import json
import string

# --- Configuration ---
SWEEP_JSON_PATH = "results/Llama-3.2-3B-Instruct_IF_sweep_20260723_170740.json"
OUTPUT_JSON_PATH = "auto_formatted_tasks_per_prompt.json"
MODEL_KEY = "llama3b"
DROP_THRESHOLD = 0.7  # accuracy_drop is a fraction, so this is a 70% drop


# Purely instructional words that should never be the Intentional Frame
STOPWORDS = {
    "a", "an", "the", "in", "is", "for", "to", "of", "with", "that", "this",
    "word", "provide", "produce", "identify", "generate", "given", "write",
    "find", "what", "sentence", "text", "please", "output", "return", "below",
    "translate", "convert", "change", "following", "use", "using","which","extract","not","select",
}


def clean_word(w):
    """Lowercases and removes punctuation from a word."""
    return w.lower().strip(string.punctuation)


def extract_best_if_for_prompt(prompt_results, threshold):
    """
    Analyzes a single prompt's results and returns (best_if, source, candidates).

    `source` is "threshold" when the IF cleared `threshold`, "fallback" when it
    came from the highest-drop non-stopword, and None when no IF could be found.
    `candidates` is the ranked list of (phrase, drop) that cleared the threshold.
    """
    # Index by the word's position in the sentence (find_if.py sorts by drop)
    by_index = {r["index"]: r for r in prompt_results["results"]}
    results = [by_index[i] for i in sorted(by_index)]

    high_impact_indices = []

    # 1. Find all indices of words that cause a meaningful accuracy drop
    for res in results:
        w = clean_word(res["element"])
        if res["accuracy_drop"] >= threshold and w and w not in STOPWORDS:
            high_impact_indices.append(res["index"])

    phrases = []
    if high_impact_indices:
        # 2. Group adjacent indices to capture multi-word IFs (like "title case")
        current_phrase = [high_impact_indices[0]]
        for idx in high_impact_indices[1:]:
            if idx == current_phrase[-1] + 1:
                current_phrase.append(idx)
            else:
                phrases.append(current_phrase)
                current_phrase = [idx]
        phrases.append(current_phrase)

    # 3. Score the reconstructed phrases
    phrase_scores = []
    for phrase_idx_list in phrases:
        # The score of a multi-word phrase is the max drop of its constituent words
        phrase_drop = max(by_index[i]["accuracy_drop"] for i in phrase_idx_list)
        phrase_str = " ".join(clean_word(by_index[i]["element"]) for i in phrase_idx_list)
        phrase_scores.append((phrase_str, phrase_drop))

    if phrase_scores:
        phrase_scores.sort(key=lambda x: x[1], reverse=True)
        return phrase_scores[0][0], "threshold", phrase_scores

    # 4. FALLBACK: If nothing passed the threshold (or baseline was 0),
    # we MUST still return a meaningful IF for this prompt.
    # We rank all non-stopwords by drop, then by word length (to prefer substantive words).
    fallback_words = []
    for r in results:
        cw = clean_word(r["element"])
        if cw and cw not in STOPWORDS:
            fallback_words.append((cw, r["accuracy_drop"], len(cw)))

    if fallback_words:
        # Sort by drop (desc), then word length (desc)
        fallback_words.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return fallback_words[0][0], "fallback", []

    return None, None, []


def generate_formatted_tasks(json_path, model_key, threshold):
    with open(json_path, 'r') as f:
        sweep_data = json.load(f)

    formatted_tasks = {}

    for task, task_info in sweep_data["results"].items():
        if "prompts" not in task_info or not task_info["prompts"]:
            formatted_tasks[task] = {model_key: [], "omit": True}
            continue

        prompt_entries = []

        # Iterate through EVERY prompt tested for this task, keeping them separate
        for prompt_idx, prompt_data in enumerate(task_info["prompts"]):
            best_if, source, candidates = extract_best_if_for_prompt(prompt_data, threshold)

            prompt_entries.append({
                "prompt_idx": prompt_idx,
                # prompt_text is the reliable join key: prompt ordering depends on
                # the selection run, so match on text rather than on index.
                "prompt_text": prompt_data.get("prompt_text"),
                "baseline_accuracy": prompt_data.get("baseline_accuracy"),
                # list-of-one, so it can be passed straight to lrp.attribution
                "ifs": [best_if] if best_if else [],
                "source": source,
                "candidates": [{"phrase": p, "drop": d} for p, d in candidates],
            })

        formatted_tasks[task] = {
            model_key: prompt_entries,
            "omit": all(not e["ifs"] for e in prompt_entries),
        }

    return formatted_tasks


def lookup_ifs(formatted_tasks, task, model_key, prompt_text, default=None):
    """Helper for consumers: get the IF list for one instruction of one task."""
    for entry in formatted_tasks.get(task, {}).get(model_key, []):
        if entry["prompt_text"] == prompt_text:
            return entry["ifs"] or ([default] if default else [])
    return [default] if default else []


if __name__ == "__main__":
    print(f"Reading sweep data from {SWEEP_JSON_PATH}...")

    auto_dict = generate_formatted_tasks(SWEEP_JSON_PATH, MODEL_KEY, DROP_THRESHOLD)

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(auto_dict, f, indent=4)

    n_prompts = sum(len(cfg[MODEL_KEY]) for cfg in auto_dict.values())
    n_threshold = sum(1 for cfg in auto_dict.values()
                      for e in cfg[MODEL_KEY] if e["source"] == "threshold")
    print(f"Successfully saved IF configuration for {len(auto_dict)} tasks "
          f"({n_prompts} prompts, {n_threshold} above threshold) to {OUTPUT_JSON_PATH}")

    print("\nPreview of extracted configurations:")
    for task, config in list(auto_dict.items())[:5]:
        print(f"  {task}:")
        for entry in config[MODEL_KEY]:
            print(f"    P{entry['prompt_idx'] + 1} [{entry['source']}] {entry['ifs']}"
                  f"  <- {entry['prompt_text']!r}")