import json
import sys
from pathlib import Path

# ---- HQ mapping (you can expand this easily) ----
ORG_HQ_COUNTRY = {
    "OpenAI": "USA",
    "DeepMind (Google)": "United Kingdom",
    "Google DeepMind": "USA",
    "Anthropic": "USA",
    "Cohere": "Canada",
    "Mistral AI": "France",
    "xAI": "USA",
    "Meta AI (FAIR)": "USA",
    "Microsoft AI (MSR)": "USA",
    "AI21 Labs": "Israel",
    "Baidu (AI/Research)": "China",
    "Qwen (Alibaba DAMO)": "China",
    "DeepSeek": "China",
    "TII (Falcon)": "United Arab Emirates",
    "01.AI (Yi)": "China",
    "InternLM (Shanghai AI Lab)": "China",
    "XVERSE": "China",
    "MosaicML (Databricks)": "USA",
    "NVIDIA Research": "USA",
    "Upstage": "South Korea",
    "rinna": "Japan",
    "EleutherAI": "USA",
    "Stability AI": "United Kingdom",
    "THUDM (GLM)": "China",
    "Baichuan": "China",
    "OpenBMB": "China",
}

# ---- main logic ----
def main(input_path: str):
    input_file = Path(input_path)
    output_file = input_file.with_name(input_file.stem + ".country_hq.json")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for record in data:
        org = record.get("display_org", "").strip()
        country = ORG_HQ_COUNTRY.get(org)
        record["country_hq"] = country if country else "Unknown"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Added 'country_hq' to {len(data)} records.")
    print(f"💾 Saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_country_hq.py <path_to_json>")
        sys.exit(1)
    main(sys.argv[1])
