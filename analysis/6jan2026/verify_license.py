import json
from collections import Counter

DATA_PATH = r"C:\Users\spatt\Desktop\opentointerpretation\huggingface\hf_6jan2026.json"

SELECTED_ORGS = {
    "OpenAI": ["OpenAI"],
    "Anthropic": ["Anthropic"],
    "Google DeepMind": ["Google DeepMind"],
    "xAI": ["xAI"],
    "Meta": ["Meta AI (FAIR)"],
    "Mistral": ["Mistral AI"],
    "Cohere": ["Cohere"],
    "Baidu": ["Baidu"],
    "Qwen (Alibaba)": ["Qwen (Alibaba DAMO)"],
    "DeepSeek": ["DeepSeek"],
}

def normalize_org_name(display_org):
    for standard_name, variants in SELECTED_ORGS.items():
        if display_org in variants:
            return standard_name
    return None

def apply_license_corrections(record):
    org = record.get("normalized_org", "")
    license_val = record.get("license")
    
    if org == "DeepSeek" and license_val == "other":
        record["license"] = "mit"
    
    if org == "xAI" and (license_val is None or license_val == ""):
        record["license"] = "bespoke"
    
    return record

def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Filter and correct
    filtered_data = []
    for record in data:
        org = record.get("display_org", "")
        normalized = normalize_org_name(org)
        if normalized:
            record_copy = record.copy()
            record_copy["normalized_org"] = normalized
            record_copy = apply_license_corrections(record_copy)
            filtered_data.append(record_copy)
    
    print("="*70)
    print("LICENSE VERIFICATION")
    print("="*70)
    
    # Count all unique licenses AFTER corrections
    all_licenses = Counter()
    for record in filtered_data:
        lic = record.get("license")
        if lic is None:
            all_licenses["<None>"] += 1
        else:
            all_licenses[str(lic)] += 1
    
    print(f"\nAll unique license values after corrections:")
    print("-"*70)
    for lic, count in sorted(all_licenses.items(), key=lambda x: -x[1]):
        print(f"  {lic}: {count}")
    
    # Verify categorization
    print(f"\n{'='*70}")
    print("CATEGORIZATION CHECK")
    print("="*70)
    
    standard_licenses = ["apache-2.0", "mit", "cc-by-4.0", "cc-by-nc-4.0"]
    
    should_be_bespoke = []
    should_be_none = []
    should_be_standard = []
    
    for lic, count in all_licenses.items():
        if lic == "<None>":
            should_be_none.append((lic, count))
        elif lic in standard_licenses:
            should_be_standard.append((lic, count))
        else:
            should_be_bespoke.append((lic, count))
    
    print(f"\nStandard licenses ({sum(c for _, c in should_be_standard)} total):")
    for lic, count in should_be_standard:
        print(f"  {lic}: {count}")
    
    print(f"\nBespoke licenses ({sum(c for _, c in should_be_bespoke)} total):")
    for lic, count in sorted(should_be_bespoke, key=lambda x: -x[1]):
        print(f"  {lic}: {count}")
    
    print(f"\nNone/Missing ({sum(c for _, c in should_be_none)} total):")
    for lic, count in should_be_none:
        print(f"  {lic}: {count}")
    
    print(f"\n{'='*70}")
    total_standard = sum(c for _, c in should_be_standard)
    total_bespoke = sum(c for _, c in should_be_bespoke)
    total_none = sum(c for _, c in should_be_none)
    grand_total = total_standard + total_bespoke + total_none
    
    print(f"TOTALS:")
    print(f"  Standard: {total_standard} ({total_standard/grand_total*100:.1f}%)")
    print(f"  Bespoke: {total_bespoke} ({total_bespoke/grand_total*100:.1f}%)")
    print(f"  None: {total_none} ({total_none/grand_total*100:.1f}%)")
    print(f"  TOTAL: {grand_total}")

if __name__ == "__main__":
    main()