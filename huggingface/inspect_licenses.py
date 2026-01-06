import json
import os

# Path to your dataset
DATA_PATH = r"C:\Users\spatt\Desktop\opentointerpretation\huggingface\hf_modelcard.licensed_v3.country_hq.cleaned.json"

# Optional output for large entries
OUTPUT_PATH = r"C:\Users\spatt\Desktop\opentointerpretation\huggingface\hf_large_entries.json"

# Threshold in characters for what counts as a "large" entry
SIZE_THRESHOLD = 1000

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a list of JSON objects.")

    large_entries = []
    license_families = {}

    for entry in data:
        entry_str = json.dumps(entry)
        if len(entry_str) > SIZE_THRESHOLD:
            large_entries.append(entry)

        lf = entry.get("license_family")
        if lf:
            license_families[lf] = license_families.get(lf, 0) + 1
        else:
            license_families["(missing)"] = license_families.get("(missing)", 0) + 1

    # Summary
    print("\n=== License Family Summary ===")
    for k, v in license_families.items():
        print(f"{k}: {v}")

    print(f"\nTotal entries: {len(data)}")
    print(f"Large entries (> {SIZE_THRESHOLD} chars): {len(large_entries)}")

    # Write large entries to a new file
    if large_entries:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
            json.dump(large_entries, out, indent=2, ensure_ascii=False)
        print(f"\nLarge entries written to: {OUTPUT_PATH}")
    else:
        print("\nNo large entries found.")

if __name__ == "__main__":
    main()
