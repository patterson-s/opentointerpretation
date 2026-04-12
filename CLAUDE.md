# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Research project analyzing open-source licensing practices of frontier AI labs on HuggingFace. The pipeline collects model metadata for ~26 orgs, enriches license fields via multi-fallback resolution, adds org HQ country, and produces tabular reports for analysis.

## Environment Setup

```bash
pip install -r requirements.txt
# requirements: huggingface_hub, beautifulsoup4, requests, lxml, jupyter, ipykernel, matplotlib, pandas
```

## Key Commands

### Fetch models from HuggingFace
```bash
# Unified pipeline (fetch + enrich + country in one pass)
python huggingface/hf_pipeline.py --output huggingface/hf_models_enriched.json

# Legacy fetch-only (no enrichment)
python huggingface/hf_full.py --with-metadata --out huggingface/hf_global_org_model_cards.json
```

### Enrich licenses (standalone)
```bash
python huggingface/enrich_hf_licenses.py <input.json> <output.json> [--interval 100]
```

### Add country HQ field
```bash
python huggingface/add_country_hq.py <input.json>
# Output: <input>.country_hq.json
```

### Run analysis reports
```bash
# Country × license matrix (counts + proportions)
python analysis/license_country_report.py huggingface/hf_modelcard.licensed_v3.country_hq.json

# Org-specific breakdowns (per-analyzer scripts in analysis/6jan2026/)
python analysis/6jan2026/license_analysis.py
python analysis/6jan2026/analyze_openai.py
# Other org analyzers: analyze_google.py, analyze_xai.py, analyze_mistral.py, etc.
```

## Architecture

### Data flow
```
hf_full.py / hf_pipeline.py
    → fetch model list per org via huggingface_hub.list_models()
    → enrich_hf_licenses.py (multi-fallback license resolution)
    → add_country_hq.py (add country_hq field)
    → analysis/ scripts (reporting)
```

### License resolution fallback chain (enrich_hf_licenses.py)
1. HF API (`/api/models/<id>`) — direct `license` field, `cardData`, tags
2. Static HTML scrape of model page
3. Hydration JSON in `<script>` tags
4. Raw README.md YAML front matter
5. LICENSE file fingerprint matching

For `license == "other"`: additional resolution via README body links, hydration URLs, and TERMS/EULA sibling files — maps to vendor slugs like `proprietary:nvidia-aiml`.

### License overrides
`hf_pipeline.py` contains two override dictionaries:
- `MODEL_LICENSE_OVERRIDES` — per-model-id corrections (AI21, NVIDIA, Qwen 1.5, etc.)
- `NV_NVCLV1`, `NV_OTHER`, `QWEN_*` sets — org-level fallback assignments
- `NVIDIA_ORG_FALLBACK` and `QWEN_ORG_FALLBACK` — catch-all for remaining unresolved models

### Org handle mapping
Both `hf_full.py` and `hf_pipeline.py` define `ORG_HANDLES` / `DEFAULT_ORG_HANDLES` dicts mapping display names (e.g. `"Mistral AI"`) to HF org handles (e.g. `["mistralai", "mistral"]`). Multiple handles are tried in order; first non-empty result wins.

### Data files
- `huggingface/hf_6jan2026.json` — snapshot used by `analysis/6jan2026/` scripts (hardcoded paths)
- `huggingface/hf_modelcard.licensed_v3.country_hq.json` — enriched + country dataset
- `huggingface/hf_modelcard.licensed_v3.country_hq.cleaned.json` — cleaned version
- `closed/` — version history JSONs for closed-source models (OpenAI, Claude, Gemini)

### Analysis scripts
`analysis/6jan2026/` contains per-org inspection scripts (all reference `hf_6jan2026.json` via hardcoded `DATA_PATH`). `license_analysis.py` is the main cross-org report for the 10 focal orgs. `analysis/license_country_report.py` is a reusable CLI tool for any enriched JSON.

### License categorization logic (analysis/6jan2026/license_analysis.py)
- Standard open: `apache-2.0`, `mit`, `cc-by-4.0`, `cc-by-nc-4.0`
- Everything else → `bespoke` (company-specific or custom)
- Org-specific corrections: DeepSeek `other` → `mit`; xAI `null` → `bespoke`
