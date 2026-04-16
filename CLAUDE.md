# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Research project analyzing open-source licensing practices of frontier AI labs on HuggingFace. The pipeline collects model metadata for ~26 orgs, enriches license fields via multi-fallback resolution, adds org HQ country, loads everything into PostgreSQL, and exposes the data through a REST API with a browser-based explorer.

**Current direction:** continuous collection is implemented — `huggingface/collect.py` appends new models incrementally on each run, and the web UI displays the last collection date via `GET /api/status`.

## Environment Setup

```bash
pip install -r requirements.txt
```

PostgreSQL database `opentointerpretation` (local, `postgres` user, no password). Credentials in `.env`:
```
PGHOST=localhost  PGPORT=5432  PGDATABASE=opentointerpretation  PGUSER=postgres  PGPASSWORD=
```

## Key Commands

### Run the web app / API
```bash
run.bat                                        # Windows: starts uvicorn on http://localhost:8080
uvicorn api.app:app --reload --port 8080       # equivalent manual command
# API docs: http://localhost:8080/docs
# NOTE: port 8000 is held by the Claude Code app and cannot be used
```

### Database
```bash
python db/ingest.py --sample        # insert first 5 records per source, print rows
python db/ingest.py                 # full ingestion (4,258 records total)
# Source file: huggingface/hf_6jan2026_meta.json (enriched; do NOT revert to hf_6jan2026.json)

# Apply schema migrations (in order for a fresh DB)
psql -U postgres -d opentointerpretation -f db/migrate_add_license_text.sql
psql -U postgres -d opentointerpretation -f db/migrate_add_collection_runs.sql
psql -U postgres -d opentointerpretation -f db/migrate_research_tables.sql
psql -U postgres -d opentointerpretation -f db/migrate_add_geocoords.sql
psql -U postgres -d opentointerpretation -f db/migrate_add_model_card.sql

# Update existing rows with new metadata fields (after re-running enrich_hf_metadata.py)
python db/update_metadata.py --input huggingface/hf_6jan2026_meta_v2.json
python db/update_metadata.py --input huggingface/hf_6jan2026_meta_v2.json --dry-run  # preview

# Fetch raw model card text (README.md) for all HF models into model_card_raw column
python huggingface/fetch_model_cards.py               # all missing rows (~7h for 72K models)
python huggingface/fetch_model_cards.py --limit 10    # test first 10
python huggingface/fetch_model_cards.py --model-id meta-llama/Llama-2-7b-hf  # single model
python huggingface/fetch_model_cards.py --force       # re-fetch all
```

### Research pipeline (company office locations)
Requires `SERPER_API_KEY` and `COHERE_API_KEY` in `.env`. Run DB migrations above first.
```bash
python research/pipeline.py                          # all 10 companies
python research/pipeline.py --companies "OpenAI"     # single company (must match companies.display_name)
python research/pipeline.py --max-sources 5 --dry-run  # preview without writing

python research/geocode.py              # geocode all findings missing coordinates (Nominatim/OSM)
python research/geocode.py --force      # re-geocode all rows
python research/geocode.py --dry-run    # print without writing
```

### Incremental collection (continuous)
```bash
# Run this to add any new HF models not yet in the DB (license + metadata enriched)
python huggingface/collect.py              # full run; inserts new models directly to DB
python huggingface/collect.py --sample     # count new models without inserting
python huggingface/collect.py --limit 10   # test with first 10 new models
python huggingface/collect.py --dry-run    # preview without writing to DB
# Requires: db/migrate_add_collection_runs.sql applied first
# API status: GET /api/status → {last_collected_at, new_models_added, total_models_in_db}
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

# Fetch license texts into the licenses table (idempotent; skips already-populated rows)
python huggingface/fetch_license_texts.py             # all missing slugs
python huggingface/fetch_license_texts.py --slug mit  # single slug
python huggingface/fetch_license_texts.py --force     # re-fetch all
# SERPER_API_KEY env var required for 4 slugs without canonical URLs
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
- `api/routers/licenses.py` — `GET /api/licenses` (list with model counts) and `GET /api/licenses/{slug}` (detail with full text + companies using it)
- `api/routers/models.py` — three endpoints:
  - `GET /api/models/filters` — distinct countries and modalities for dropdown population (must be registered before the path route)
  - `GET /api/models` — paginated (50/page) filtered list; query params: `company_id`, `license_slug`, `country_hq`, `data_source`, `modality`, `limit`, `offset`
  - `GET /api/models/{model_id:path}` — full detail; uses `:path` type to capture slashes in HuggingFace model IDs (e.g. `01-ai/Yi-1.5-34B`)
- `api/routers/analysis.py` — aggregation endpoints:
  - `GET /api/analysis/model-releases-by-country` / `model-releases-by-company`
  - `GET /api/analysis/historical-total` / `historical-by-company` / `historical-by-country` — monthly releases using `models.release_date`; flat rows that JS pivots into Chart.js datasets
- `api/routers/status.py` — `GET /api/status` → `{last_collected_at, new_models_added, total_models_seen, total_models_in_db, status}` from the most recent `collection_runs` row
- `api/routers/research.py` — research provenance endpoints:
  - `GET /api/research/map` — all geocoded findings for map visualization (lat/lng populated)
  - `GET /api/research/companies/{company_id}/sources` — sources fetched for a company
  - `GET /api/research/companies/{company_id}/findings` — extracted + fact-checked locations
  - `GET /api/research/sources/{source_id}/chunks` — text chunks for a source (embeddings excluded)
  - `POST /api/research/query` — ad-hoc RAG query; body `{"company_id": int, "question": str}`; uses cosine retrieval → Cohere rerank → Command-A answer

**Adding a new section:** create `api/routers/<name>.py`, include it in `api/app.py`, add `'<name>'` to the `SECTIONS` array in `web/app.js`, add `<section id="section-<name>">` in `web/index.html`, and a nav link. The hash router in `activateSection()` handles the rest automatically.

### Frontend (`web/`)
- `web/index.html` — SPA shell; nav tabs hash-route to `#companies`, `#models`, `#countries`, `#licenses`, `#analysis`
- `web/app.js` — all section logic; no build step
- `web/style.css` — no framework; CSS custom properties in `:root` for colors/fonts

**Section layout patterns:**

*Two-panel (Companies, Licenses):* `.company-list-panel` (240px, scrollable list) + `.company-detail-panel` (flex: 1, detail view). Adding a new two-panel section requires only `flex-direction: row` on the section ID — the shared panel classes handle everything else.

*Filter-table-detail (Models):* `.models-filter-panel` (220px sidebar with `<select>` dropdowns) + `.models-main-panel` (flex: 1). The main panel hosts two mutually exclusive views toggled by JS: `#models-table-view` (paginated table) and `#models-detail-view` (detail with back button). Filters lazy-load on first section activation via `modelsLoaded` flag.

*Sub-tab + sub-nav + chart (Analysis):* `#section-analysis` uses `flex-direction: column`. A `.analysis-subtab-bar` sits at the top with two `.analysis-subtab` buttons — **Snapshot** and **Historical**. Each button shows/hides a `.analysis-subpanel` div. Inside each sub-panel is a `.analysis-container` with a 240px `.analysis-nav` left pane, `.analysis-content` center, and `.analysis-notes` bottom strip. `#historical` hash redirects to `#analysis` and activates the Historical sub-tab. Init is guarded by `analysisNavReady` and `historicalNavReady` flags; `setupAnalysisSection()` coordinates both.

**Key JS patterns:**
- `renderBarChart(container, data, labelField, title)` — reusable vertical bar chart; destroys `activeAnalysisChart` before rendering
- `pivotHistoricalData(rows, keyField, topN)` — converts flat `[{month, key, count}]` rows into Chart.js multi-dataset format for stacked charts
- `loadNotesForMetric(metric)` / `loadHistoricalNotes(metric)` — localStorage persistence keyed to metric name
- License slug links in company detail and model detail navigate to `#licenses` and select the slug via `selectLicense()`
- Section init guards (`modelsLoaded`, `analysisNavReady`, `historicalNavReady`) prevent duplicate setup on repeated hash changes
- `renderModelDetail(container, d)` — shared function used by both the Models section and the Companies tab model detail view
- `loadCompanyModelDetail(modelId, returnPanel)` — renders model detail inline inside `#company-detail` with a back button that restores the saved company HTML
- `groupModelsByPrefix(models)` — clusters a company's model list into a typed render-tree (`group → subgroup → row`) using longest-common-token-prefix matching; singletons remain flat rows
- `stripOrgPrefix(modelId)` / `tokenizeName(name)` — helpers used by the grouping logic

**Companies tab detail panel structure:**
1. Header + stat cards (country, model count, HF count, closed count)
2. License Distribution — plain three-column table (License / Count / %) with no chart
3. Models — grouped table of up to 50 models rendered by `renderCompanyModelsTable()`:
   - `tr.model-group-header` — series name spanning all columns (e.g. `AceMath`, `Falcon`)
   - `tr.model-subgroup-header` — sub-series name (e.g. `H1`, `E-1B`), indented 1.5rem
   - `tr.model-variant-row` — variant suffix only (e.g. `7B-Instruct`), indented under its group; `data-model-id` still holds full ID for click-to-detail
   - Singletons (no shared prefix) rendered as flat rows with org prefix stripped
   - Clicking any row calls `loadCompanyModelDetail()` which opens full detail inline

### Research package (`research/`)
Three-phase pipeline for AI company office location research:

1. **Phase 1 — search & embed** (`pipeline.py` → `serper_client.py` + `cohere_client.py`): runs Serper queries, fetches pages, chunks text (~500 chars, 50-char overlap), embeds with `embed-v4.0` (1024-dim), stores to `research_sources` + `research_chunks`.
2. **Phase 2 — extract** (`rag.py:extract_locations()`): cosine retrieval → Cohere `rerank-v4.0-pro` → Prompt 1 to Command-A (`command-a-03-2025`) with JSON mode → list of `{finding_type, city, country}`.
3. **Phase 3 — fact-check** (`rag.py:fact_check()`): per candidate, fetches 3 extra substantiation sources, re-runs RAG → Prompt 2 → `{confirmed: bool, notes: str}` → upserts into `research_findings`.

Embeddings are stored as PostgreSQL `FLOAT[]` (not pgvector). Cosine similarity is computed in Python (numpy). No pgvector extension required.

`geocode.py` — post-pipeline step; queries Nominatim (1 req/sec OSM rate limit) to populate `latitude`/`longitude` on `research_findings`. Skips rows where city starts with "Various" or is a regional placeholder.

### Database schema (`db/schema.sql`)
Three tables: `companies` → `models` ← `licenses`
- `models.data_source` — `'huggingface'` | `'openai'` | `'anthropic'` | `'google'`
- `models.metadata JSONB` — enriched HF fields: `pipeline_tag`, `modality`, `num_parameters`, `last_modified`, `library_name`, `gated`, `language`, `architectures`, `model_type`, `tags`
- `models.release_date` (DATE) populated from HF `createdAt`; 99.9% coverage after enrichment
- `models.license_id` nullable (22.6% null rate)
- `licenses` columns: `slug`, `display_name`, `family`, `is_osi_approved`, `notes`, `license_text`, `source_url`, `allows_commercial_use`, `allows_derivatives`, `requires_attribution`, `requires_share_alike`
- Schema migrations in `db/migrate_*.sql` — always use `ADD COLUMN IF NOT EXISTS` / `CREATE TABLE IF NOT EXISTS`
- Research tables (added via migrations): `research_sources` → `research_chunks` (cascades on delete); `research_findings` (unique on `company_id, finding_type, city, country`); `research_finding_sources` junction table

### HuggingFace enrichment scripts
All scripts write a `.ckpt.json` checkpoint on each interval for resume support.

**`enrich_hf_licenses.py`** — 5-layer license fallback chain:
1. HF REST API (`/api/models/{id}`)
2. Static HTML scrape of model page
3. Hydration JSON in `<script>` tags
4. Raw README.md YAML front matter
5. LICENSE file fingerprint matching

For `license == "other"`: additional resolution via README body links and TERMS/EULA sibling files — maps to vendor slugs like `proprietary:nvidia-aiml`.

**`enrich_hf_metadata.py`** — fetches `/api/models/{id}` per model and extracts `pipeline_tag`, derived `modality`, `num_parameters` in billions (from `safetensors.parameters`, not bytes), `release_date`, `last_modified`, `library_name`, `gated`, `language`, `architectures`, `model_type`. Also extracts: `base_model`, `fine_tuned_from`, `datasets`, `eval_metrics`, `trending_score`, `inference`, `vocab_size`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `max_position_embeddings`, `safetensors_total`, `has_license_file`, `has_readme`, `disabled`, `private`.

**`fetch_model_cards.py`** — DB-direct script (like `fetch_license_texts.py`) that fetches the raw README.md for each HF model and writes it to `models.model_card_raw`. `model_card_fetched_at` serves as the checkpoint column — rows with NULL are fetched on each run.

**`db/update_metadata.py`** — merges new metadata fields into existing rows using PostgreSQL JSONB `||` operator. Use after re-running `enrich_hf_metadata.py` to update an existing DB without full re-ingestion. Additive by default (existing keys not overwritten); pass `--overwrite` to replace.

### License overrides (`huggingface/hf_pipeline.py`)
- `MODEL_LICENSE_OVERRIDES` — per-model-id corrections (AI21, NVIDIA, Qwen 1.5, etc.)
- `NVIDIA_ORG_FALLBACK` / `QWEN_ORG_FALLBACK` — catch-all for remaining unresolved models

### License text pipeline (`huggingface/fetch_license_texts.py`)
- `CANONICAL_URLS` maps each slug to a direct fetch URL; `SERPER_SEARCH_HINTS` for unusual slugs
- Skips `other` and `unknown` slugs; 27/32 slugs populated; 4 require `SERPER_API_KEY` (hybrid/NVIDIA slugs)

### Analysis scripts (`analysis/6jan2026/`)
All reference `huggingface/hf_6jan2026.json` via hardcoded `DATA_PATH`. License categorization: `apache-2.0`, `mit`, `cc-by-4.0`, `cc-by-nc-4.0` → open; everything else → `bespoke`.

### Data files
- `huggingface/hf_6jan2026.json` — base snapshot (8 fields/model); input to enrichment scripts and `analysis/6jan2026/` scripts
- `huggingface/hf_6jan2026_meta.json` — enriched snapshot (19 fields/model); **canonical input to `db/ingest.py`**
- `huggingface/hf_modelcard.licensed_v3.country_hq.json` — enriched + country dataset (license-focused)
- `closed/` — version history JSONs for OpenAI, Claude, Gemini
