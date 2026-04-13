# Project Log

## 13 April 2026 (continued, session 3)

### License Text Storage and Licenses Tab

**Context**
Each open-weight model on HuggingFace carries a license slug (e.g. `apache-2.0`, `tongyi-qianwen`, `llama3`). The database already had a `licenses` table with slugs but no actual license text. Goal: store the full text of each unique license, add analysis boolean columns, expose licenses through the API, and add a dedicated Licenses tab to the web explorer.

**Database migration (`db/migrate_add_license_text.sql`)**
Six new columns added to the `licenses` table:
- `license_text TEXT` — full text of the license
- `source_url VARCHAR(1000)` — URL the text was fetched from
- `allows_commercial_use BOOLEAN`
- `allows_derivatives BOOLEAN`
- `requires_attribution BOOLEAN`
- `requires_share_alike BOOLEAN`

**License text fetcher (`huggingface/fetch_license_texts.py`)**
- Iterates all license slugs in the DB; skips already-populated rows (idempotent)
- `CANONICAL_URLS` dict maps each of the 32 slugs to a known source: Apache.org, Creative Commons, GitHub raw (Meta Llama, BigScience, NVIDIA, Qwen, AI21), Google AI dev (Gemma), HuggingFace blog (OpenRAIL), and AI21's website (Jamba)
- Falls back to Serper API search for hybrid/unusual slugs when `SERPER_API_KEY` is set
- CLI flags: `--dry-run`, `--slug SLUG`, `--force`
- **Result: 27/32 slugs populated** (4 require Serper API key; `other`/`unknown` intentionally skipped)

**API (`api/routers/licenses.py`)**
- `GET /api/licenses` — all licenses sorted by model count descending; includes all analysis boolean columns
- `GET /api/licenses/{slug}` — full detail: license text, source URL, companies using this license with per-company model lists; 404 on unknown slug
- Registered in `api/app.py`

**Frontend**

*Nav and section:* Licenses tab added between Countries and Analysis. Uses the same two-panel layout as Companies (reuses `.company-list-panel` / `.company-detail-panel` CSS classes with zero new layout CSS).

*Left panel:* Searchable list of all 32 license slugs, sorted by model count. Supports `display_name` fallback label.

*Right panel (license detail):*
- Header with slug, display name, family badge (color-coded: open-source/bespoke/proprietary/unknown), OSI Approved badge
- Four stat cards: Commercial Use, Derivatives, Attribution, Share-Alike (Yes / No / — from boolean columns)
- Companies using this license with model counts
- Full license text in a scrollable dark `<pre>` block, or "not yet fetched" placeholder
- "View source ↗" link to the source URL

*Company detail click-through:* The license distribution in the Companies tab now renders as a clickable table in addition to the bar chart. Clicking any license slug navigates to `#licenses` and selects that slug.

**Port change**
Port 8000 was found to be held by a non-killable system-level process (PID visible in netstat but not in Win32 process list — likely the Claude Code app). `run.bat` updated to use **port 8080**. CORS origins updated accordingly. App now runs at `http://localhost:8080`.

---

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
