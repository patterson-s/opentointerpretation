import json
from pathlib import Path
from collections import Counter, defaultdict

DATA_PATH = r"C:\Users\spatt\Desktop\opentointerpretation\huggingface\hf_6jan2026.json"

# Selected organizations mapping: display name -> HF org names
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
    """Map dataset org names to standardized names"""
    for standard_name, variants in SELECTED_ORGS.items():
        if display_org in variants:
            return standard_name
    return None

def apply_license_corrections(record):
    """Apply license corrections based on organization-specific rules"""
    org = record.get("normalized_org", "")
    license_val = record.get("license")
    
    # DeepSeek: all "other" → "mit"
    if org == "DeepSeek" and license_val == "other":
        record["license"] = "mit"
    
    # xAI: None → "bespoke" 
    if org == "xAI" and (license_val is None or license_val == ""):
        record["license"] = "bespoke"
    
    return record

def categorize_license(license_val):
    """Categorize licenses into simplified groups"""
    # Handle None/missing values
    if license_val is None or license_val == "" or license_val == "Unknown":
        return "None"
    
    # Convert to string if needed
    license_str = str(license_val).strip()
    
    # Standard open licenses (keep as-is)
    if license_str in ["apache-2.0", "mit", "cc-by-4.0", "cc-by-nc-4.0"]:
        return license_str
    
    # Everything else is bespoke (company-specific)
    # This includes: gemma, llama variants, qwen/tongyi variants, hybrid, other, etc.
    return "bespoke"

def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Filter to selected organizations
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
    print("FILTERED DATASET - 10 SELECTED ORGANIZATIONS")
    print("="*70)
    print(f"Total models: {len(filtered_data)}")
    print(f"Organizations: {len(SELECTED_ORGS)}")
    
    # Count unique license types
    all_licenses = set(r.get("license") for r in filtered_data if r.get("license"))
    print(f"Unique license types: {len(all_licenses)}")
    
    # Overall license distribution
    print(f"\nOverall License Distribution:")
    print("-"*70)
    license_counts = Counter(r.get("license", "Unknown") for r in filtered_data)
    for lic, count in license_counts.most_common():
        pct = (count / len(filtered_data)) * 100
        print(f"  {lic}: {count} ({pct:.1f}%)")
    
    # Simplified proportional breakdown
    print(f"\n{'='*70}")
    print("SIMPLIFIED LICENSE CATEGORIES")
    print("="*70)
    simplified_counts = Counter()
    bespoke_details = Counter()
    
    for record in filtered_data:
        lic = record.get("license", "Unknown")
        category = categorize_license(lic)
        simplified_counts[category] += 1
        
        # Track what goes into bespoke
        if category == "bespoke":
            bespoke_details[lic] += 1
    
    print("\nProportional Breakdown:")
    print("-"*70)
    for category, count in simplified_counts.most_common():
        pct = (count / len(filtered_data)) * 100
        print(f"  {category}: {count} ({pct:.1f}%)")
    
    print(f"\nBespoke licenses include:")
    print("-"*70)
    for lic, count in sorted(bespoke_details.items(), key=lambda x: -x[1]):
        pct = (count / simplified_counts["bespoke"]) * 100
        print(f"  {lic}: {count} ({pct:.1f}% of bespoke)")
    
    # Organization breakdown
    print(f"\n{'='*70}")
    print("BY ORGANIZATION")
    print("="*70)
    
    orgs = defaultdict(list)
    for r in filtered_data:
        org = r.get("normalized_org", "Unknown")
        orgs[org].append(r)
    
    # Sort by country grouping then name
    country_order = {
        "OpenAI": 1, "Anthropic": 2, "Google DeepMind": 3, "xAI": 4, "Meta": 5,
        "Mistral": 6, "Cohere": 7,
        "Baidu": 8, "Qwen (Alibaba)": 9, "DeepSeek": 10
    }
    
    for org in sorted(orgs.keys(), key=lambda x: country_order.get(x, 99)):
        models = orgs[org]
        print(f"\n{org}: {len(models)} models")
        print("-"*70)
        
        # Detailed licenses
        org_licenses = Counter(r.get("license", "Unknown") for r in models)
        for lic, count in org_licenses.most_common():
            pct = (count / len(models)) * 100
            print(f"  {lic}: {count} ({pct:.1f}%)")
        
        # Simplified categories for this org
        org_simplified = Counter(categorize_license(r.get("license", "Unknown")) for r in models)
        print(f"\n  Simplified:")
        for category, count in org_simplified.most_common():
            pct = (count / len(models)) * 100
            print(f"    {category}: {count} ({pct:.1f}%)")
    
    # License type summary
    print(f"\n{'='*70}")
    print("UNIQUE LICENSE TYPES")
    print("="*70)
    for lic in sorted(all_licenses):
        print(f"  - {lic}")
    
    # Check for missing organizations
    print(f"\n{'='*70}")
    print("ORGANIZATION VERIFICATION")
    print("="*70)
    found_orgs = set(orgs.keys())
    expected_orgs = set(SELECTED_ORGS.keys())
    missing_orgs = expected_orgs - found_orgs
    
    if missing_orgs:
        print(f"⚠️  Missing organizations (no models found):")
        for org in sorted(missing_orgs):
            print(f"  - {org}")
    else:
        print("✓ All 10 organizations found")

if __name__ == "__main__":
    main()