import json
from pathlib import Path
from collections import Counter

DATA_PATH = r"C:\Users\spatt\Desktop\opentointerpretation\huggingface\hf_6jan2026.json"

def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    meta_models = [r for r in data if r.get("display_org") == "Meta AI (FAIR)"]
    
    print(f"Meta AI (FAIR) Models: {len(meta_models)}")
    print(f"\nLicense Breakdown:")
    print("-" * 50)
    
    licenses = Counter(r.get("license", "Unknown") for r in meta_models)
    for lic, count in licenses.most_common():
        print(f"  {lic}: {count}")
    
    print(f"\nTotal: {sum(licenses.values())}")
    
    print(f"\nAll Meta AI (FAIR) Models:")
    print("-" * 50)
    for model in sorted(meta_models, key=lambda x: x.get("model_id", "")):
        model_id = model.get("model_id", "Unknown")
        license = model.get("license", "Unknown")
        print(f"  {model_id} [{license}]")

if __name__ == "__main__":
    main()