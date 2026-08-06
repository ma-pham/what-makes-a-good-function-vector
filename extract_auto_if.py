import argparse
import json
import os
import string

# --- Configuration ---
# One output file holds every model's keyword lists, keyed by the model's `if_key`
# (see conf/model/*.yaml), so a run for one model MERGES into it rather than
# replacing it. Defaults keep the original single-model behaviour.
SWEEP_JSON_PATH = "results/Llama-3.2-3B-Instruct_IF_sweep_20260804_023241.json"
OUTPUT_JSON_PATH = "auto_formatted_tasks2.json"
MODEL_KEY = "llama3b"
DROP_THRESHOLD = 0.7  # 5% accuracy drop required to be "high impact"


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
    Analyzes a single prompt's results and returns its best Intentional Frame.
    Handles multi-word IFs (e.g., "title case") by grouping adjacent high-impact words.
    """
    # Ensure results are strictly ordered by their index in the sentence
    results = sorted(prompt_results["results"], key=lambda x: x["index"])
    
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
        phrase_drop = max([results[i]["accuracy_drop"] for i in phrase_idx_list])
        phrase_str = " ".join([clean_word(results[i]["element"]) for i in phrase_idx_list])
        phrase_scores.append((phrase_str, phrase_drop))

    # Return the highest-scoring phrase if one exists
    if phrase_scores:
        phrase_scores.sort(key=lambda x: x[1], reverse=True)
        return phrase_scores[0][0]

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
        return fallback_words[0][0]
        
    return None


def generate_formatted_tasks(json_path, model_key, threshold):
    with open(json_path, 'r') as f:
        sweep_data = json.load(f)

    formatted_tasks = {}

    for task, task_info in sweep_data["results"].items():
        if "prompts" not in task_info or not task_info["prompts"]:
            formatted_tasks[task] = {model_key: [], "omit": True}
            continue

        task_ifs = set()

        # Iterate through EVERY prompt tested for this task
        for global_prompt_idx, prompt_data in enumerate(task_info["prompts"]):
            best_if = extract_best_if_for_prompt(prompt_data, threshold)
            if best_if:
                task_ifs.add(best_if)

        # Convert the set of all prompt IFs to a list
        final_ifs = list(task_ifs)

        formatted_tasks[task] = {
            model_key: final_ifs,
            "omit": len(final_ifs) == 0
        }

    return formatted_tasks


def merge_into(path, auto_dict, model_key):
    """Write `auto_dict`'s lists under `model_key`, keeping every other model's entries.

    Returns (merged, added, updated). "omit" is shared across models in this schema, so
    a disagreement is reported rather than silently overwritten.
    """
    merged = {}
    if os.path.exists(path):
        with open(path) as f:
            merged = json.load(f)
        kept = sorted({k for t in merged.values() if isinstance(t, dict) for k in t} - {model_key, "omit"})
        print(f"Merging into existing {path}: {len(merged)} tasks, keeping keys {kept or ['(none)']}")

    added = updated = 0
    for task, cfg in auto_dict.items():
        entry = merged.setdefault(task, {})
        updated += model_key in entry
        added += model_key not in entry
        if "omit" in entry and entry["omit"] != cfg["omit"]:
            print(f"  note: {task!r} omit differs (kept {entry['omit']}, this run says {cfg['omit']})")
        else:
            entry["omit"] = cfg["omit"]
        entry[model_key] = cfg[model_key]

    with open(path, "w") as f:
        json.dump(merged, f, indent=4)
    return merged, added, updated


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__ or "Turn an IF sweep into keyword lists.")
    ap.add_argument("--sweep", default=SWEEP_JSON_PATH, help="results/{model}_IF_sweep_{ts}.json from find_if.py")
    ap.add_argument("--model-key", default=MODEL_KEY, help="must match if_key in conf/model/*.yaml")
    ap.add_argument("--out", default=OUTPUT_JSON_PATH)
    ap.add_argument("--threshold", type=float, default=DROP_THRESHOLD)
    args = ap.parse_args()
    SWEEP_JSON_PATH, MODEL_KEY, OUTPUT_JSON_PATH, DROP_THRESHOLD = (
        args.sweep, args.model_key, args.out, args.threshold)

    print(f"Reading sweep data from {SWEEP_JSON_PATH}...")
    auto_dict = generate_formatted_tasks(SWEEP_JSON_PATH, MODEL_KEY, DROP_THRESHOLD)
    merged, added, updated = merge_into(OUTPUT_JSON_PATH, auto_dict, MODEL_KEY)
    print(f"Saved {MODEL_KEY}: {added} task(s) added, {updated} updated "
          f"-> {OUTPUT_JSON_PATH} now holds {len(merged)} tasks")
    
    # Print a preview
    print("\nPreview of extracted configurations:")
    for task, config in list(auto_dict.items())[:5]:
        print(f"  {task}: {config[MODEL_KEY]}")