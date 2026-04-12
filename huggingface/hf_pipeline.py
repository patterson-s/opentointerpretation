#!/usr/bin/env python
import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests
from huggingface_hub import list_models, ModelInfo

ORG_HANDLES: Dict[str, List[str]] = {
    "Mistral AI": ["mistralai", "mistral"],
    "Cohere": ["CohereLabs", "Cohere"],
    "OpenAI": ["openai"],
    "Google DeepMind": ["google"],
    "Perplexity": ["perplexity-ai"],
    "Baidu": ["baidu"],
    "Qwen (Alibaba DAMO)": ["Qwen"],
    "DeepSeek": ["deepseek-ai"],
    "Microsoft AI (MSR)": ["microsoft"],
    "xAI": ["xai-org"],
    "Meta AI (FAIR)": ["meta-llama"],
    "NVIDIA Research": ["nvidia"],
    "AI21 Labs": ["ai21labs"],
    "TII (Falcon)": ["tiiuae"],
    "01.AI (Yi)": ["01-ai"],
    "THUDM (GLM)": ["THUDM"],
    "Baichuan": ["baichuan-inc"],
    "InternLM (Shanghai AI Lab)": ["InternLM"],
    "XVERSE": ["xverse"],
    "OpenBMB": ["openbmb"],
    "EleutherAI": ["EleutherAI"],
    "MosaicML (Databricks)": ["mosaicml"],
    "Upstage": ["upstage"],
    "rinna": ["rinna"],
    "Stability AI": ["stabilityai"],
    "Anthropic": ["anthropic"],
}

ORG_HQ_COUNTRY = {
    "OpenAI": "USA",
    "Google DeepMind": "USA",
    "Anthropic": "USA",
    "Cohere": "Canada",
    "Mistral AI": "France",
    "xAI": "USA",
    "Meta AI (FAIR)": "USA",
    "Microsoft AI (MSR)": "USA",
    "AI21 Labs": "Israel",
    "Baidu": "China",
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

HF_API = "https://huggingface.co/api/models/"
HTTP_TIMEOUT = 20
UA = {"User-Agent": "opentointerpretation-license-bot/3.0"}

LICENSE_SLUG_MAP = {
    "apache 2.0": "apache-2.0",
    "apache-2": "apache-2.0",
    "apache-2.0": "apache-2.0",
    "mit": "mit",
    "bsd-3-clause": "bsd-3-clause",
    "bsd-2-clause": "bsd-2-clause",
    "mpl-2.0": "mpl-2.0",
    "agpl-3.0": "agpl-3.0",
    "gpl-3.0": "gpl-3.0",
    "lgpl-3.0": "lgpl-3.0",
}

LICENSE_FINGERPRINTS = [
    (re.compile(r"apache\s+license\s+version\s*2\.0", re.I), "apache-2.0"),
    (re.compile(r"\bmit\s+license\b", re.I), "mit"),
    (re.compile(r"\bmozilla public license\s*version\s*2\.0\b", re.I), "mpl-2.0"),
    (re.compile(r"\bgnu\s+general\s+public\s+license\s+version\s*3\b", re.I), "gpl-3.0"),
    (re.compile(r"\bgnu\s+lesser\s+general\s+public\s+license\s+version\s*3\b", re.I), "lgpl-3.0"),
    (re.compile(r"\baffero\s+general\s+public\s+license\s+version\s*3\b", re.I), "agpl-3.0"),
    (re.compile(r"\bbsd\s*(?:3|three)[-\s]*clause\b", re.I), "bsd-3-clause"),
]

MODEL_LICENSE_OVERRIDES = {
    "ai21labs/AI21-Jamba-Large-1.5": "jamba-open-model-license",
    "ai21labs/AI21-Jamba-Large-1.6": "jamba-open-model-license",
    "ai21labs/AI21-Jamba-Large-1.7": "jamba-open-model-license",
    "ai21labs/AI21-Jamba-Large-1.7-FP8": "jamba-open-model-license",
    "ai21labs/AI21-Jamba-Mini-1.5": "jamba-open-model-license",
    "ai21labs/AI21-Jamba-Mini-1.7": "jamba-open-model-license",
    "ai21labs/AI21-Jamba-Mini-1.7-FP8": "jamba-open-model-license",
    "baidu/Qianfan-VL-3B": "hybrid_mit_qwen",
    "baidu/Qianfan-VL-70B": "hybrid_MIT_metallama3.1commnuity",
    "baidu/Qianfan-VL-8B": "hybrid_MIT_metallama3.1commnuity",
    "nvidia/audio_to_audio_schrodinger_bridge": "nvidia-oneway-noncommercial-license",
    "nvidia/VideoITG-8B": "nvlicense",
    "nvidia/Nemotron-H-4B-Base-8K": "nvidia-internal-scientific-research-and-development-model-license",
    "nvidia/Nemotron-H-4B-Instruct-128K": "nvidia-internal-scientific-research-and-development-model-license",
    "nvidia/RADIO": "nvidia-source-code-license",
    "nvidia/RADIO-B": "nvidia-source-code-license",
    "nvidia/RADIO-g": "nvidia-source-code-license",
    "nvidia/RADIO-H": "nvidia-source-code-license",
    "nvidia/RADIO-L": "nvidia-source-code-license",
}

NV_NVCLV1 = {
    "nvidia/cascade_mask_rcnn_mamba_vision_base_3x_coco",
    "nvidia/cascade_mask_rcnn_mamba_vision_small_3x_coco",
    "nvidia/cascade_mask_rcnn_mamba_vision_tiny_3x_coco",
    "nvidia/mamba_vision_160k_ade20k-512x512_base",
    "nvidia/mamba_vision_160k_ade20k-512x512_small",
    "nvidia/mamba_vision_160k_ade20k-512x512_tiny",
    "nvidia/mamba_vision_160k_ade20k-640x640_l3_21k",
    "nvidia/MambaVision-L-1K",
    "nvidia/MambaVision-L2-1K",
    "nvidia/MambaVision-L2-512-21K",
    "nvidia/MambaVision-L3-256-21K",
    "nvidia/MambaVision-S-1K",
    "nvidia/MambaVision-T2-1K",
}
for mid in NV_NVCLV1:
    MODEL_LICENSE_OVERRIDES[mid] = "nvclv1"

NV_OTHER = {
    "nvidia/mit-b0", "nvidia/mit-b1", "nvidia/mit-b4", "nvidia/mit-b5",
    "nvidia/segformer-b0-finetuned-ade-512-512",
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "nvidia/segformer-b0-finetuned-cityscapes-640-1280",
    "nvidia/segformer-b0-finetuned-cityscapes-768-768",
    "nvidia/segformer-b1-finetuned-ade-512-512",
    "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
    "nvidia/segformer-b2-finetuned-ade-512-512",
    "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "nvidia/segformer-b3-finetuned-ade-512-512",
    "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    "nvidia/segformer-b4-finetuned-ade-512-512",
    "nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
    "nvidia/segformer-b5-finetuned-ade-640-640",
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}
for mid in NV_OTHER:
    MODEL_LICENSE_OVERRIDES[mid] = "other"

QWEN_1_5_MODELS = {
    "Qwen/Qwen1.5-1.8B-Chat-GGUF",
    "Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4",
    "Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8",
    "Qwen/Qwen1.5-32B",
    "Qwen/Qwen1.5-4B",
    "Qwen/Qwen1.5-4B-Chat",
    "Qwen/Qwen1.5-4B-Chat-AWQ",
    "Qwen/Qwen1.5-4B-Chat-GGUF",
    "Qwen/Qwen1.5-4B-Chat-GPTQ-Int4",
    "Qwen/Qwen1.5-4B-Chat-GPTQ-Int8",
}
for mid in QWEN_1_5_MODELS:
    MODEL_LICENSE_OVERRIDES[mid] = "tongyi-qianwen-research"

QWEN_2_VL_72B = {"Qwen/Qwen2-VL-72B"}
for mid in QWEN_2_VL_72B:
    MODEL_LICENSE_OVERRIDES[mid] = "qwen"

QWEN_2_5_3B_MODELS = {
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct-AWQ",
    "Qwen/Qwen2.5-3B-Instruct-GGUF",
    "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
    "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct-AWQ",
    "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
    "Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4",
    "Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int8",
    "Qwen/Qwen2.5-Omni-3B",
}
for mid in QWEN_2_5_3B_MODELS:
    MODEL_LICENSE_OVERRIDES[mid] = "qwen-research"

NVIDIA_ORG_FALLBACK = "nvidia-open-model-license"
QWEN_ORG_FALLBACK = "tongyi-qianwen"


def fetch_models(display_org: str, handles: List[str]) -> List[Dict[str, Any]]:
    for handle in handles:
        try:
            infos = list(list_models(author=handle, cardData=False, fetch_config=False, full=False))
            if not infos:
                continue
            records = []
            for m in infos:
                mid = getattr(m, "modelId", getattr(m, "id", None))
                if not mid:
                    continue
                records.append({
                    "display_org": display_org,
                    "handle": handle,
                    "model_id": mid,
                    "url": f"https://huggingface.co/{mid}",
                    "license": getattr(m, "license", None),
                    "likes": getattr(m, "likes", None),
                    "downloads": getattr(m, "downloads", None),
                })
            dedup = {r["model_id"]: r for r in records}
            return list(dedup.values())
        except Exception as e:
            print(f"[WARN] {display_org} ({handle}): {e}")
            continue
    return []


def normalize_license(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    slug = re.sub(r"\s+", " ", s.strip()).lower()
    slug = slug.replace("_", "-")
    slug = LICENSE_SLUG_MAP.get(slug, slug)
    slug = re.sub(r"apache\s*(?:licen[cs]e)?\s*version?\s*2(?:\.0)?", "apache-2.0", slug)
    slug = re.sub(r"\bmit(?=\b)", "mit", slug)
    slug = re.sub(r"\bmpl\s*2(?:\.0)?\b", "mpl-2.0", slug)
    return slug


def get_http(url: str, **kw) -> requests.Response:
    return requests.get(url, timeout=HTTP_TIMEOUT, headers=UA, **kw)


def license_from_api(repo_id: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        r = get_http(HF_API + repo_id)
        if r.status_code != 200:
            return None, f"API {r.status_code}"
        data = r.json()
        cand = data.get("license") or (data.get("cardData") or {}).get("license") or (data.get("metadata") or {}).get("license")
        if cand:
            return normalize_license(cand), None
        tags = data.get("tags") or []
        for t in tags:
            m = re.match(r"license\s*:\s*(.+)", str(t), re.I)
            if m:
                return normalize_license(m.group(1)), None
        return None, "API miss"
    except Exception as e:
        return None, f"API error: {e}"


def license_from_readme(repo_id: str) -> Tuple[Optional[str], Optional[str]]:
    url = f"https://huggingface.co/{repo_id}/raw/README.md"
    try:
        r = get_http(url)
        if r.status_code != 200:
            return None, f"readme {r.status_code}"
        m = re.match(r"\s*---\s*\r?\n(.*?)\r?\n---\s*\r?\n", r.text, re.S)
        if not m:
            return None, "readme miss: no front matter"
        fm = m.group(1)
        for line in fm.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                if k.strip().lower() == "license":
                    return normalize_license(v.strip().strip('"').strip("'")), None
        return None, "readme miss: no license"
    except Exception as e:
        return None, f"readme error: {e}"


def license_from_license_file(repo_id: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        r = get_http(HF_API + repo_id)
        if r.status_code != 200:
            return None, f"licensefile api {r.status_code}"
        siblings = (r.json() or {}).get("siblings") or []
        lic_name = None
        for sib in siblings:
            name = (sib.get("rfilename") or "").strip()
            if name.upper() in {"LICENSE", "LICENSE.TXT"}:
                lic_name = name
                break
        if not lic_name:
            return None, "licensefile miss: no LICENSE blob"
        raw = f"https://huggingface.co/{repo_id}/resolve/main/{lic_name}"
        rr = get_http(raw, allow_redirects=True)
        if rr.status_code != 200:
            return None, f"licensefile fetch {rr.status_code}"
        txt = rr.text or ""
        for pat, slug in LICENSE_FINGERPRINTS:
            if pat.search(txt):
                return slug, None
        return None, "licensefile miss: unknown text"
    except Exception as e:
        return None, f"licensefile error: {e}"


def enrich_license(rec: Dict[str, Any]) -> Dict[str, Any]:
    repo_id = rec.get("model_id")
    current = (rec.get("license") or "").strip()
    
    if not current:
        lic, note = license_from_api(repo_id)
        if not lic:
            lic, note = license_from_readme(repo_id)
        if not lic:
            lic, note = license_from_license_file(repo_id)
        if lic:
            rec["license"] = lic
            rec["_license_source"] = note
    
    return rec


def apply_manual_corrections(rec: Dict[str, Any]) -> Dict[str, Any]:
    model_id = rec.get("model_id", "")
    org = rec.get("display_org", "")
    
    if model_id in MODEL_LICENSE_OVERRIDES:
        rec["license"] = MODEL_LICENSE_OVERRIDES[model_id]
        return rec
    
    if org == "NVIDIA Research":
        rec["license"] = NVIDIA_ORG_FALLBACK
        return rec
    
    if org == "Qwen (Alibaba DAMO)":
        rec["license"] = QWEN_ORG_FALLBACK
        return rec
    
    license_url = rec.get("license_url", "")
    if isinstance(license_url, str) and "apache-2.0" in license_url.lower():
        rec["license"] = "apache-2.0"
    
    return rec


def add_country(rec: Dict[str, Any]) -> Dict[str, Any]:
    org = rec.get("display_org", "").strip()
    rec["country_hq"] = ORG_HQ_COUNTRY.get(org, "Unknown")
    return rec


def main():
    parser = argparse.ArgumentParser(description="Fetch and enrich HuggingFace models")
    parser.add_argument("--orgs", help="Comma-separated list of organizations (defaults to all)")
    parser.add_argument("--output", default="hf_models_enriched.json", help="Output file path")
    args = parser.parse_args()
    
    if args.orgs:
        org_names = [o.strip() for o in args.orgs.split(",")]
        selected_orgs = {k: v for k, v in ORG_HANDLES.items() if k in org_names}
        if not selected_orgs:
            print(f"No matching organizations found. Available: {', '.join(ORG_HANDLES.keys())}")
            return
    else:
        selected_orgs = ORG_HANDLES
    
    print(f"Fetching models for {len(selected_orgs)} organizations...")
    
    all_records = {}
    for display_org, handles in selected_orgs.items():
        records = fetch_models(display_org, handles)
        if records:
            print(f"  {display_org}: {len(records)} models")
            for r in records:
                all_records[r["model_id"]] = r
        else:
            print(f"  {display_org}: no models found")
    
    print(f"\nEnriching {len(all_records)} unique models...")
    enriched = []
    for i, rec in enumerate(all_records.values()):
        rec = enrich_license(rec)
        rec = apply_manual_corrections(rec)
        rec = add_country(rec)
        
        keep_fields = {"display_org", "handle", "model_id", "url", "license", "country_hq", "likes", "downloads"}
        clean_rec = {k: rec.get(k) for k in keep_fields if k in rec}
        enriched.append(clean_rec)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}...")
    
    enriched.sort(key=lambda x: (x["display_org"].lower(), x["model_id"].lower()))
    
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    
    print(f"\nComplete. Wrote {len(enriched)} records to: {output_path}")


if __name__ == "__main__":
    main()