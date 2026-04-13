# Project Log

## 13 April 2026 (continued)

### Historical Model Releases Tab

**New tab: Historical**
- Added a fourth analysis tab (`#historical`) to the top nav alongside Companies, Models, Countries, and Analysis.
- Follows the same 3-pane layout as the Analysis tab (left sub-nav, center chart, right notes).
- Fixed a pre-existing stray `</div></section>` in `web/index.html`.

**Three sub-views:**
- **Total** — filled line chart of all model releases by month (2018–2026)
- **By Company** — stacked bar chart, top 10 companies by total release count, one color per company
- **By Country** — stacked bar chart, all 9 countries, one color per country

**Backend (`api/routers/analysis.py`)** — three new endpoints added to the existing analysis router:
- `GET /api/analysis/historical-total` — monthly totals using `models.release_date`
- `GET /api/analysis/historical-by-company` — flat rows pivoted by JS into per-company datasets
- `GET /api/analysis/historical-by-country` — same pivot approach by `country_hq`

**Frontend (`web/app.js`)**
- `pivotHistoricalData()` — converts flat `[{month, key, count}]` rows into Chart.js multi-dataset format; supports optional top-N filter
- `renderHistoricalLineChart()` — line chart with fill for total view
- `renderHistoricalStackedChart()` — stacked bar chart with 10-color palette and legend
- `loadHistoricalNotes()` — localStorage notes scoped to `historical-notes-{metric}`
- `setupHistoricalNavigation()` — guarded click-delegation (no duplicate listener attachment on hash changes)

**Database backfill**
- `db/ingest.py` updated to use `hf_6jan2026_meta.json` (enriched) as the HuggingFace source.
- HF records had been ingested without `release_date` (base JSON had no date field); ran a targeted `UPDATE` to backfill `release_date` and `metadata` for all 4,210 HF rows from the meta JSON.

**Dataset summary after full ingestion:**
- 52 months of data (2018-06 to 2026-02)
- Peak: March 2022 — 679 models
- 24 companies, 9 countries

---

## 13 April 2026

### Analysis Dashboard Implementation with Mistral Vibe

**Added Analysis Tab and Metrics System**
- Added new "Analysis" tab to the web interface navigation
- Implemented hash-based routing for the analysis section (`#analysis`)
- Created dedicated analysis section in `web/index.html` with metric navigation

**Implemented Model Release Metrics**
- Added two core analysis metrics:
  - **Model Releases by Country**: Bar chart showing model distribution across countries
  - **Model Releases by Company**: Bar chart showing model distribution across companies
- Created unified chart rendering function with dynamic labeling
- Added proper titles and styling for each visualization

**Built Persistent Notes System**
- Added notes section below each analysis visualization
- Implemented localStorage-based persistence for notes
- Notes are automatically saved on input and persist between sessions
- Each metric maintains its own separate notes
- Clean UI with monospace header and resizable textarea

**Backend API Enhancements**
- Created `api/routers/analysis.py` with four endpoints:
  - `GET /api/analysis/model-releases-by-country`
  - `GET /api/analysis/model-releases-by-company`
  - `GET /api/analysis/license-trends` (placeholder)
  - `GET /api/analysis/country-comparison` (placeholder)
- Integrated analysis router into main FastAPI app
- Optimized SQL queries for each metric type

**Frontend JavaScript Enhancements**
- Extended section router to include 'analysis' in SECTIONS array
- Added comprehensive analysis navigation setup with event listeners
- Implemented dynamic metric loading with error handling
- Added debug logging for development
- Created reusable chart rendering function

**UI/UX Improvements**
- Added analysis-specific CSS styles
- Created navigation menu for metric selection
- Implemented active state highlighting
- Added loading and error states
- Maintained consistent design language with existing interface

**Technical Implementation Details**
- Used Chart.js for interactive bar chart visualizations
- Implemented localStorage for client-side note persistence
- Added proper error handling and user feedback
- Maintained code organization and separation of concerns
- All changes follow existing project patterns and conventions

**Verification & Testing**
- Both metrics load and display correctly
- Notes persist across page refreshes
- Navigation between metrics works smoothly
- Error states handled gracefully
- Responsive design maintained

**Development Process**
- This implementation was completed with Mistral Vibe assistance
- Used structured task management with todo tracking
- Followed iterative development with testing at each stage
- Maintained clean git commit history with proper co-author attribution

---

### HuggingFace Model Metadata Enrichment

**Context**
The existing pipeline collected only 8 fields per model (model_id, url, license, likes, downloads, display_org, handle, country_hq). The HF REST API exposes significantly more per-model data, and the database already had nullable columns for `release_date` and `metadata JSONB` that were unpopulated.

**New script: `huggingface/enrich_hf_metadata.py`**
- Fetches `https://huggingface.co/api/models/{model_id}` for each record
- Extracts and merges 11 new fields per model:
  - `pipeline_tag` — HF task type (e.g. `"text-generation"`)
  - `modality` — derived category: `"text"`, `"multimodal"`, `"vision"`, `"audio"`, `"video"`
  - `num_parameters` — exact parameter count in billions from `safetensors.parameters` (e.g. `34.39`)
  - `release_date` — repo creation date from `createdAt`
  - `last_modified` — most recent push timestamp
  - `library_name` — ML framework (e.g. `"transformers"`)
  - `gated` — access control status (`false` or `"manual"`)
  - `language` — list of supported language codes from card metadata
  - `architectures` — model architecture class (e.g. `["LlamaForCausalLM"]`)
  - `model_type` — model family string (e.g. `"llama"`)
  - `tags` — filtered tag list (arxiv IDs, region, task tags)
- Checkpoint recovery: writes `.ckpt.json` every N records; resumes on restart
- CLI: `--input`, `--output`, `--interval`, `--delay`, `--limit`

**Updated: `db/ingest.py`**
- Added `safe_date()` helper to parse ISO 8601 timestamps to Python `date`
- `ingest_hf()` now populates `release_date` and `metadata` JSONB columns from enriched records
- No schema changes required — both columns already existed

**Run**
```bash
# Test (10 records)
python huggingface/enrich_hf_metadata.py --input huggingface/hf_6jan2026.json --output huggingface/hf_6jan2026_meta.json --limit 10

# Full run (~4,200 models, ~25 min at 0.3s/call)
python huggingface/enrich_hf_metadata.py --input huggingface/hf_6jan2026.json --output huggingface/hf_6jan2026_meta.json
```

## 12 April 2026

### Session summary

**Initialized project documentation**
- Created `CLAUDE.md` describing the project purpose, key commands, and pipeline architecture for future Claude Code sessions.

**Built PostgreSQL database**
- Designed a three-table relational schema (`companies`, `licenses`, `models`) to organize the existing flat JSON data.
- Created `db/schema.sql` with all `CREATE TABLE` and index statements.
- Created `db/ingest.py` — an ETL script with `--sample` and full-run modes, NULL-safe helpers, per-record error handling, and progress logging.
- Created `.env.example` as a credentials template.
- Created and connected to the `opentointerpretation` PostgreSQL database (local, `postgres` user, no password).
- Applied the schema and ran full ingestion.

**Ingestion results**

| Table | Rows |
|---|---|
| `companies` | 24 |
| `licenses` | 32 |
| `models` | 4,258 |

Sources loaded:
- `huggingface/hf_6jan2026.json` — 4,210 HuggingFace models across 23 orgs
- `closed/openai_gpt_version_history.json` — 22 OpenAI models
- `closed/claude_version_history.json` — 15 Anthropic models
- `closed/gemini_version_history.json` — 11 Google DeepMind models

Verification passed: zero orphaned foreign keys, zero null `model_id` values. 1,003 models have a null `license_id`, consistent with the 22.6% null rate in the source HF data and the absence of license fields in closed-source version histories.

---

**Built REST API and web explorer**

- Added `fastapi`, `uvicorn[standard]`, `psycopg2-binary`, and `python-dotenv` to `requirements.txt`.
- Created `api/db.py` — psycopg2 connection helper using `.env` credentials; yields a dict-mode cursor via context manager.
- Created `api/routers/companies.py` — two endpoints:
  - `GET /api/companies` — list all 24 companies (id, display_name, country_hq, hf_handle), sorted alphabetically.
  - `GET /api/companies/{id}` — full company detail: country HQ, model count, HuggingFace vs. closed-source breakdown, license distribution.
- Created `api/app.py` — FastAPI app; includes companies router; mounts `web/` as static files at `/`; CORS middleware for localhost dev.
- Created `web/index.html` — single-page app shell with top nav tabs (Companies, Models, Countries).
- Created `web/style.css` — clean research aesthetic: monospace accents, neutral gray palette, blue accent (`#2563eb`), no external CSS framework.
- Created `web/app.js` — hash-based section router (`#companies`, `#models`, `#countries`); searchable company list; company detail panel with stat cards and a Chart.js horizontal bar chart for license distribution.

**Design notes**
- Models and Countries tabs render "Coming soon" placeholders — structure is in place for future viewers.
- License bar chart color-codes known open-source slugs (apache-2.0, mit, etc.) in accent blue; unknown/bespoke in gray.
- New sections or company detail panels require only localized additions (new route file + HTML section); no structural refactor needed.

**Verification**
- `GET /api/companies` → 24 companies returned.
- `GET /api/companies/1` → 01.AI: 28 models, all apache-2.0.
- Frontend served at `http://localhost:8000`.
- Run: `uvicorn api.app:app --reload` from project root.
