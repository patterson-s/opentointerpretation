# get_global_org_models.py
from __future__ import annotations
import argparse
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple
from huggingface_hub import list_models, ModelInfo

# Display name -> list of candidate HF handles (first successful one is used)
DEFAULT_ORG_HANDLES: Dict[str, List[str]] = {
    # You already had these:
    "Mistral": ["mistralai", "mistral"],             # "mistralai" is correct
    "Cohere": ["CohereLabs", "Cohere"],
    "OpenAI": ["openai"],
    "Google": ["google"],
    "Perplexity": ["perplexity-ai"],
    "Baidu": ["baidu"],
    "Qwen": ["Qwen"],                                # capital Q on HF
    "DeepSeek": ["deepseek-ai"],

    # You asked to add:
    "Microsoft": ["microsoft"],
    "xAI": ["xai-org"],

    # Strong global coverage (open/frontier labs & families):
    "Meta (Llama)": ["meta-llama"],
    "NVIDIA": ["nvidia"],
    "AI21 Labs": ["ai21labs"],
    "TII UAE (Falcon)": ["tiiuae"],
    "01.AI (Yi)": ["01-ai"],
    "THUDM (GLM)": ["THUDM"],
    "Baichuan": ["baichuan-inc"],
    "InternLM (Shanghai AI Lab)": ["InternLM"],
    "XVERSE": ["xverse"],
    "OpenBMB (BAAI)": ["openbmb"],
    "EleutherAI": ["EleutherAI"],
    "MosaicML (Databricks)": ["mosaicml"],
    "Upstage": ["upstage"],
    "rinna": ["rinna"],

    # Optional/completeness (often more vision/multimodal or sparse LLM):
    "Stability AI": ["stabilityai"],

    # Might be sparse / uncertain handles; we’ll warn if empty:
    "Tencent Hunyuan": ["HunyuanOpen", "TencentARC"],
    "Huawei (Pangu/Noah)": ["huaweinoah", "HuaweiNOAH"],
}

def as_simple_record(display_org: str, handle: str, m: ModelInfo) -> Dict[str, Any]:
    mid = getattr(m, "modelId", getattr(m, "id", None))
    return {
        "display_org": display_org,
        "handle": handle,
        "model_id": mid,
        "url": f"https://huggingface.co/{mid}",
    }

def as_metadata_record(display_org: str, handle: str, m: ModelInfo) -> Dict[str, Any]:
    mid = getattr(m, "modelId", getattr(m, "id", None))
    return {
        "display_org": display_org,
        "handle": handle,
        "model_id": mid,
        "url": f"https://huggingface.co/{mid}",
        "private": getattr(m, "private", None),
        "gated": getattr(m, "gated", None),
        "library_name": getattr(m, "library_name", None),
        "pipeline_tag": getattr(m, "pipeline_tag", None),
        "license": getattr(m, "license", None),
        "likes": getattr(m, "likes", None),
        "downloads": getattr(m, "downloads", None),
        "tags": getattr(m, "tags", None),
        "last_modified": getattr(m, "lastModified", None),
        "sha": getattr(m, "sha", None),
    }

def fetch_by_handle(handle: str, include_private: bool) -> List[ModelInfo]:
    return list_models(
        author=handle,
        cardData=False,
        fetch_config=False,
        full=False,
        token=True if include_private else None,  # uses your HF CLI token if logged in
    )

def fetch_for_display_org(
    display_org: str,
    handles: List[str],
    include_private: bool,
    with_metadata: bool,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Try each handle for a display org; return (chosen_handle, records).
    If no handle yields results, chosen_handle="" and records=[].
    """
    builder = as_metadata_record if with_metadata else as_simple_record
    for handle in handles:
        try:
            infos = fetch_by_handle(handle, include_private)
        except Exception as e:
            print(f"[WARN] {display_org} ({handle}): error fetching models: {e}")
            continue
        if not infos:
            continue
        # Convert to records and dedupe by model_id within the chosen handle
        recs = []
        for m in infos:
            mid = getattr(m, "modelId", getattr(m, "id", None))
            if not mid:
                continue
            recs.append(builder(display_org, handle, m))
        dedup = {r["model_id"]: r for r in recs}
        return handle, list(dedup.values())
    # No results for any handle
    return "", []

def main():
    ap = argparse.ArgumentParser(
        description="Export Hugging Face model card links for a global set of frontier orgs."
    )
    ap.add_argument(
        "--with-metadata",
        action="store_true",
        help="Include license/tags/likes/downloads/last_modified/etc.",
    )
    ap.add_argument(
        "--include-private",
        action="store_true",
        help="Include gated/private repos (requires `huggingface-cli login`).",
    )
    ap.add_argument(
        "--out",
        default="hf_global_org_model_cards.json",
        help="Combined output JSON path",
    )
    ap.add_argument(
        "--split",
        action="store_true",
        help="Also write one JSON per display org: <slug>_model_cards.json",
    )
    args = ap.parse_args()

    combined: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    total_before_dedup = 0

    for display_org, handles in DEFAULT_ORG_HANDLES.items():
        chosen_handle, records = fetch_for_display_org(
            display_org, handles, include_private=args.include_private, with_metadata=args.with_metadata
        )
        if not records:
            print(f"[MISS] No models found for '{display_org}' using handles {handles}")
            missing.append(display_org)
            continue

        total_before_dedup += len(records)

        # Optional per-org file
        if args.split:
            slug = display_org.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
            fname = f"{slug}_model_cards.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(sorted(records, key=lambda x: x["model_id"].lower()), f, ensure_ascii=False, indent=2)
            print(f"[OK] {display_org} ({chosen_handle}) -> {fname} [{len(records)}]")

        # Merge into combined (dedupe by model_id across everything; last write wins)
        for r in records:
            combined[r["model_id"]] = r

    combined_list = sorted(combined.values(), key=lambda x: (x["display_org"].lower(), x["model_id"].lower()))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(combined_list, f, ensure_ascii=False, indent=2)

    print(f"\nCombined: wrote {len(combined_list)} unique entries from {len(DEFAULT_ORG_HANDLES)} display orgs -> {args.out}")
    print(f"Collected {total_before_dedup} total entries before cross-org dedup.")
    if missing:
        print("\n[SUMMARY] Orgs with no results (check handles or API-only status):")
        for name in missing:
            print(f"  - {name}")

if __name__ == "__main__":
    main()
