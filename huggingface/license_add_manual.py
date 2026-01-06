import json
import os

# Input and output paths
INPUT_PATH = r"C:\Users\spatt\Desktop\opentointerpretation\huggingface\hf_modelcard.licensed_v3.country_hq.json"
OUTPUT_PATH = r"C:\Users\spatt\Desktop\opentointerpretation\huggingface\hf_modelcard.licensed_v3.country_hq.cleaned.json"

# Fields to retain when trimming/debloating
KEEP_FIELDS = {"display_org", "handle", "model_id", "url", "license", "country_hq"}
APACHE_MATCH = "apache-2.0"

# ---------- MODEL → LICENSE OVERRIDES ----------
MODEL_LICENSE_OVERRIDES = {
    # --- AI21 Jamba ---
    "ai21labs/AI21-Jamba-Large-1.5": "jamba-open-model-license",
    "ai21labs/AI21-Jamba-Large-1.6": "jamba-open-model-license",
    "ai21labs/AI21-Jamba-Large-1.7": "jamba-open-model-license",
    "ai21labs/AI21-Jamba-Large-1.7-FP8": "jamba-open-model-license",
    "ai21labs/AI21-Jamba-Mini-1.5": "jamba-open-model-license",
    "ai21labs/AI21-Jamba-Mini-1.7": "jamba-open-model-license",
    "ai21labs/AI21-Jamba-Mini-1.7-FP8": "jamba-open-model-license",

    # --- Baidu Qianfan ---
    "baidu/Qianfan-VL-3B": "hybrid_mit_qwen",
    "baidu/Qianfan-VL-70B": "hybrid_MIT_metallama3.1commnuity",
    "baidu/Qianfan-VL-8B": "hybrid_MIT_metallama3.1commnuity",

    # --- NVIDIA ---
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

# NVIDIA nvclv1 group
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

# NVIDIA "other" group (mit*, segformer*)
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

# QWEN groups
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

# Organization-wide fallbacks
NVIDIA_ORG_FALLBACK = "nvidia-open-model-license"
QWEN_ORG_FALLBACK = "tongyi-qianwen"

# ---------- Helper functions ----------
def is_bloated(entry: dict) -> bool:
    for key in ("license_name", "license_url"):
        val = entry.get(key)
        if isinstance(val, str) and (len(val) > 200 or "&quot;" in val):
            return True
    return False

def trim(entry: dict) -> dict:
    return {k: entry.get(k) for k in KEEP_FIELDS}

# ---------- Main logic ----------
def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []
    counts = {"override": 0, "nvidia_fb": 0, "qwen_fb": 0, "deepseek": 0, "apache": 0}

    for entry in data:
        e = dict(entry)
        model_id = e.get("model_id", "")
        org = e.get("display_org", "")
        license_url = e.get("license_url", "")

        # (A) Model-specific overrides (AI21, Baidu, NVIDIA, Qwen)
        if model_id in MODEL_LICENSE_OVERRIDES:
            e["license"] = MODEL_LICENSE_OVERRIDES[model_id]
            cleaned.append(trim(e))
            counts["override"] += 1
            continue

        # (B) NVIDIA org fallback
        if org == "NVIDIA":
            e["license"] = NVIDIA_ORG_FALLBACK
            cleaned.append(trim(e))
            counts["nvidia_fb"] += 1
            continue

        # (C) Qwen org fallback
        if org == "Qwen":
            e["license"] = QWEN_ORG_FALLBACK
            cleaned.append(trim(e))
            counts["qwen_fb"] += 1
            continue

        # (D) DeepSeek bloated rule
        if org == "DeepSeek" and is_bloated(e):
            e["license"] = "DeepSeek"
            cleaned.append(trim(e))
            counts["deepseek"] += 1
            continue

        # (E) Apache via license_url
        if isinstance(license_url, str) and APACHE_MATCH in license_url.lower():
            e["license"] = "apache-2.0"
            cleaned.append(trim(e))
            counts["apache"] += 1
            continue

        # Default: keep entry
        cleaned.append(e)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        json.dump(cleaned, out, indent=2, ensure_ascii=False)

    print(f"Total entries: {len(data)}")
    for k, v in counts.items():
        print(f"{k.replace('_',' ').title()}: {v}")
    print(f"Cleaned file written to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
