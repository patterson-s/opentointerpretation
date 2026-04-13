# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Research project analyzing open-source licensing practices of frontier AI labs on HuggingFace. The pipeline collects model metadata for ~26 orgs, enriches license fields via multi-fallback resolution, adds org HQ country, loads everything into PostgreSQL, and exposes the data through a REST API with a browser-based explorer.

## Environment Setup

```bash
pip install -r requirements.txt
# Includes: huggingface_hub, fastapi, uvicorn[standard], psycopg2-binary, python-dotenv, pandas, etc.
```

PostgreSQL database `opentointerpretation` (local, `postgres` user, no password). Credentials in `.env`:
```
PGHOST=localhost  PGPORT=5432  PGDATABASE=opentointerpretation  PGUSER=postgres  PGPASSWORD=
```

## Key Commands

### Run the web app / API
```bash
uvicorn api.app:app --reload        # http://localhost:8000
run.bat                             # Windows convenience script (same command)
# API docs auto-generated at http://localhost:8000/docs
```

### Database
```bash
python db/ingest.py --sample        # insert first 5 records per source, print rows
python db/ingest.py                 # full ingestion (4,258 records total)
# Source file: huggingface/hf_6jan2026_meta.json (enriched; do NOT revert to hf_6jan2026.json)
```

### Fetch / enrich HuggingFace data
```bash
# Unified pipeline (fetch + enrich + country in one pass)
python huggingface/hf_pipeline.py --output huggingface/hf_models_enriched.json

# Standalone license enrichment
python huggingface/enrich_hf_licenses.py <input.json> <output.json> [--interval 100]

# Add country HQ field (outputs <input>.country_hq.json)
python huggingface/add_country_hq.py <input.json>

# Enrich with model metadata (pipeline_tag, modality, num_parameters, release_date, etc.)
python huggingface/enrich_hf_metadata.py \
    --input huggingface/hf_6jan2026.json \
    --output huggingface/hf_6jan2026_meta.json \
    [--limit 10]   # omit for full run (~25 min at 0.3s/call)
```

### Analysis reports
```bash
python analysis/license_country_report.py huggingface/hf_modelcard.licensed_v3.country_hq.json
python analysis/6jan2026/license_analysis.py
python analysis/6jan2026/analyze_openai.py   # other orgs: analyze_google.py, analyze_xai.py, etc.
```

## Architecture

### Overall data flow
```
hf_pipeline.py → enrich_hf_licenses.py → add_country_hq.py ─┐
                                                               ├→ enrich_hf_metadata.py → db/ingest.py → PostgreSQL
                                                               │                                              ↓
                                                               └──────────────────────────── api/ (FastAPI) → web/ (SPA)
```

### API (`api/`)
- `api/app.py` — FastAPI app; includes routers; mounts `web/` as static files at `/` (API routes take priority)
- `api/db.py` — `get_cursor()` context manager; yields a `RealDictCursor`; loads `.env` from project root
- `api/routers/companies.py` — `GET /api/companies` (list) and `GET /api/companies/{id}` (detail with license distribution)
- `api/routers/analysis.py` — aggregation endpoints:
  - `GET /api/analysis/model-releases-by-country` — model count grouped by org HQ country
  - `GET /api/analysis/model-releases-by-company` — model count grouped by company
  - `GET /api/analysis/license-trends` — license slug distribution across all models
  - `GET /api/analysis/country-comparison` — company count + model count per country
  - `GET /api/analysis/time-analysis` — model count bucketed by month (uses `models.created_at`)
  - `GET /api/analysis/historical-total` — monthly model releases using `models.release_date`
  - `GET /api/analysis/historical-by-company` — monthly releases per company (flat rows; JS pivots)
  - `GET /api/analysis/historical-by-country` — monthly releases per country (flat rows; JS pivots)

To add a new section: create `api/routers/<name>.py`, include it in `api/app.py`, add `'<name>'` to the `SECTIONS` array in `web/app.js`, add a `<section id="section-<name>">` in `web/index.html`, and a nav link. The hash router in `activateSection()` handles everything else automatically.

### Frontend (`web/`)
- `web/index.html` — SPA shell; nav tabs hash-route to `#companies`, `#models`, `#countries`, `#analysis`, `#historical`
- `web/app.js` — `activateSection()` handles hash routing; `loadCompanyList()` / `loadCompanyDetail()` drive the company viewer; Analysis and Historical sections each have their own metric sub-nav, Chart.js charts, and a localStorage-backed notes field per metric
- `web/style.css` — no framework; CSS custom properties in `:root` for colors/fonts

Key JS patterns:
- `renderBarChart(container, data, labelField, title)` — reusable vertical bar chart; destroys `activeAnalysisChart` before rendering
- `pivotHistoricalData(rows, keyField, topN)` — converts flat `[{month, key, count}]` API rows into Chart.js multi-dataset format for stacked charts
- `loadNotesForMetric(metric)` / `loadHistoricalNotes(metric)` — localStorage persistence keyed to metric name
- All analysis/historical sub-navs use click delegation on `.analysis-metrics-list` with `data-metric` attributes

### Database schema (`db/schema.sql`)
Three tables: `companies` → `models` ← `licenses`
- `models.data_source` distinguishes `'huggingface'` from `'openai'`/`'anthropic'`/`'google'`
- `models.metadata JSONB` holds enriched HF fields: `pipeline_tag`, `modality`, `num_parameters`, `last_modified`, `library_name`, `gated`, `language`, `architectures`, `model_type`, `tags`
- `models.release_date` (DATE) is populated from `createdAt` (HF repo creation date); 99.9% coverage after enrichment
- `models.license_id` is nullable (22.6% null rate from source data)

### HuggingFace enrichment scripts
All scripts support checkpoint recovery — they write a `.ckpt.json` on each interval so long runs can be resumed.

**`enrich_hf_licenses.py`** — 5-layer license fallback chain:
1. HF REST API (`/api/models/{id}`)
2. Static HTML scrape of model page
3. Hydration JSON in `<script>` tags
4. Raw README.md YAML front matter
5. LICENSE file fingerprint matching

For `license == "other"`: additional resolution via README body links and TERMS/EULA sibling files — maps to vendor slugs like `proprietary:nvidia-aiml`.

**`enrich_hf_metadata.py`** — fetches `/api/models/{id}` per model and extracts:
- `pipeline_tag` + derived `modality` category (`text` / `multimodal` / `vision` / `audio` / `video`)
- `num_parameters` in billions (from `safetensors.parameters`, which stores counts not bytes)
- `release_date` from `createdAt`, `last_modified` from `lastModified`
- `library_name`, `gated`, `language` (from `cardData`), `architectures`, `model_type` (from `config`)

### License overrides (`huggingface/hf_pipeline.py`)
- `MODEL_LICENSE_OVERRIDES` — per-model-id corrections (AI21, NVIDIA, Qwen 1.5, etc.)
- `NVIDIA_ORG_FALLBACK` / `QWEN_ORG_FALLBACK` — catch-all for remaining unresolved models

### Analysis scripts (`analysis/6jan2026/`)
All reference `huggingface/hf_6jan2026.json` via a hardcoded `DATA_PATH`. License categorization: `apache-2.0`, `mit`, `cc-by-4.0`, `cc-by-nc-4.0` → open; everything else → `bespoke`. Org-specific corrections applied in `license_analysis.py`.

### Data files
- `huggingface/hf_6jan2026.json` — base snapshot (8 fields/model); input to enrichment scripts and `analysis/6jan2026/` scripts
- `huggingface/hf_6jan2026_meta.json` — enriched snapshot (19 fields/model); **canonical input to `db/ingest.py`**
- `huggingface/hf_modelcard.licensed_v3.country_hq.json` — enriched + country dataset (license-focused)
- `closed/` — version history JSONs for OpenAI, Claude, Gemini
