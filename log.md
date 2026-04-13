# Project Log

## 13 April 2026 (continued, session 7)

### AI Company Office Research Pipeline + Map Tab

#### Research Pipeline

**Context**
Added a provenance-tracked research system to find and fact-check the headquarters and affiliate office locations of the 10 frontier AI companies in the dataset.

**Database schema additions (`db/migrate_research_tables.sql`)**
Four new tables:
- `research_sources` — raw web pages fetched during research (url, title, snippet, raw_content, search_query, source_type), affiliated to a company via `company_id`
- `research_chunks` — text chunks derived from sources, with Cohere `embed-v4.0` 1024-dim embeddings stored as `FLOAT[]` (no pgvector dependency)
- `research_findings` — extracted location findings per company (`finding_type`: `headquarters`/`affiliate`, `city`, `country`, `confidence`: `confirmed`/`unconfirmed`, `source_count`, `notes`)
- `research_finding_sources` — junction table linking findings to supporting sources

**Research pipeline (`research/pipeline.py`)**
Three-phase pipeline run per company:
1. **Search & embed** — Serper API searches (2 query variants per company), fetches up to 5 pages per query, chunks via sentence-boundary splitting, embeds with Cohere `embed-v4.0`, stores all in DB
2. **Extract locations (Prompt 1)** — retrieves chunks by cosine similarity (numpy, no pgvector), reranks with Cohere `rerank-v4.0-pro`, prompts Cohere `command-a-03-2025` with JSON mode to extract all HQ and affiliate locations
3. **Fact-check (Prompt 2)** — for each candidate, fetches up to 3 additional Serper sources, re-embeds, re-reranks, and prompts Command-A to confirm or deny the claim; marks `confirmed`/`unconfirmed`

Supporting modules:
- `research/serper_client.py` — Serper search + BeautifulSoup text extraction (mirrors `fetch_license_texts.py` pattern)
- `research/cohere_client.py` — `embed_texts()`, `embed_query()`, `rerank()`, `chunk_text()` using `ClientV2`
- `research/rag.py` — `extract_locations()` and `fact_check()` RAG functions
- All API keys loaded from `.env` via `dotenv` (not `os.environ` directly)

**Results (all 10 companies)**
77 findings total:

| Company | HQ | Confirmed offices |
|---|---|---|
| OpenAI | San Francisco, US | Brussels, Dublin, London, NYC, Paris, Seattle, Singapore, Tokyo |
| Anthropic | San Francisco, US | Bellevue, London, Munich, NYC, Paris, Seattle, Washington D.C. |
| Google DeepMind | London, UK | Montreal, Mountain View, NYC, Paris, Toronto, Zurich |
| xAI | Palo Alto, US | London, Memphis, San Francisco |
| Meta AI (FAIR) | New York City, US | London, Menlo Park, Montreal, Paris, Pittsburgh, Seattle, Tel Aviv, Zurich |
| Cohere | Toronto, Canada | London, Montreal, NYC, Palo Alto, Paris, San Francisco, Seoul |
| Mistral AI | Paris, France | London, Munich, Palo Alto |
| Baidu | Beijing, China | Hong Kong, Shanghai, Silicon Valley, Singapore, Tokyo |
| DeepSeek | Hangzhou, China | London |
| Qwen (Alibaba DAMO) | Hangzhou, China | Hong Kong |

**API endpoints added (`api/routers/research.py`)**
- `GET /api/research/map` — all geocoded findings joined with company names (used by map tab)
- `GET /api/research/companies/{id}/sources` — sources fetched for a company
- `GET /api/research/companies/{id}/findings` — extracted + fact-checked locations
- `GET /api/research/sources/{source_id}/chunks` — text chunks for a source
- `POST /api/research/query` — ad-hoc RAG query over a company's stored chunks

---

#### Geocoding (`research/geocode.py`, `db/migrate_add_geocoords.sql`)

Added `latitude FLOAT` and `longitude FLOAT` columns to `research_findings`. Geocoding script calls Nominatim (OpenStreetMap, free, no API key) at 1 req/sec. Skips entries where city starts with "Various" (Baidu regional placeholders). 73 of 77 findings geocoded successfully (4 skipped).

---

#### Map Tab (`web/index.html`, `web/app.js`, `web/style.css`)

New "Map" nav tab added as the sixth section. Uses Leaflet 1.9.4 (CDN) and topojson-client 3 (CDN).

**Features:**
- **Country choropleth** — world boundaries loaded from world-atlas TopoJSON (~99KB); countries shaded blue proportional to office density (HQ weight = 3, confirmed affiliate = 1)
- **HQ markers** — diamond shape (`L.divIcon`, rotated square), company color, white border; visually distinct from affiliates
- **Affiliate markers** — filled circles (`L.circleMarker`), company color; unconfirmed offices rendered at lower opacity with no fill
- **Arc lines** — parabolic `L.Polyline` with 80 intermediate points; bump height capped at 15° latitude (`Math.min(dist * 0.28, 15)`) so transoceanic routes stay within the visible map; confirmed arcs solid, unconfirmed dashed
- **Left sidebar** — company toggles (colored diamond swatch + checkbox) and country toggles; filter logic uses `addLayer`/`removeLayer` per item rather than `setStyle`, so it works correctly with both `L.marker` and `L.circleMarker`
- **Pan/zoom** — Leaflet built-in; `invalidateSize()` called after section activation to handle deferred initialization

**Company colors**

| Company | Color |
|---|---|
| OpenAI | `#2563eb` blue |
| Anthropic | `#7c3aed` purple |
| Google DeepMind | `#059669` green |
| xAI | `#dc2626` red |
| Meta AI (FAIR) | `#0284c7` sky |
| Cohere | `#d97706` amber |
| Mistral AI | `#db2777` pink |
| Baidu | `#65a30d` lime |
| DeepSeek | `#0891b2` cyan |
| Qwen | `#ea580c` orange |

---

## 13 April 2026 (continued, session 6)

### Companies Tab — Grouped Model Display

**Context**
The companies tab model list showed full `org/model-name` IDs in a flat table. The org prefix is redundant inside a company view, and models belonging to the same series (e.g. `aya-expanse-8b`, `aya-expanse-32b`) appeared as unrelated rows.

**Naming pattern analysis**
Examined model IDs across top orgs (tiiuae, nvidia, Qwen, CohereLabs, EleutherAI, google) to characterize nesting depth:
- Single-level: `AceMath-7B-Instruct` / `AceMath-72B-RM` → group `AceMath`, variants `7B-Instruct` / `72B-RM`
- Two-level: `Falcon-H1-1.5B-Base` / `Falcon-H1-1.5B-Instruct` / `Falcon-E-1B-Base` → group `Falcon` → subgroups `H1` / `E-1B` → variant rows
- Singletons (no shared prefix with any sibling) remain as flat rows

**Algorithm (`groupModelsByPrefix` in `web/app.js`)**
- Strip org prefix with `stripOrgPrefix()`
- Tokenize by `-` / `_` with `tokenizeName()`
- Cluster by first token using a Map; within each cluster, compute the longest shared token prefix via `longestSharedPrefix()`
- Recurse one level: within each top-level group, re-cluster the suffix strings to detect sub-groups
- Returns a typed render-tree: `group → subgroup → row` (or flat `row` for singletons)
- When a model IS the series base with no suffix, displays `(base)` as the variant label

**Rendering (`renderCompanyModelsTable` in `web/app.js`)**
- `tr.model-group-header` — spans all 5 columns, shows series name; not clickable
- `tr.model-subgroup-header` — indented sub-series header; not clickable
- `tr.model-variant-row` — same 5-column layout; first cell shows variant suffix only with 1.5rem left padding; tooltip retains full `model_id`; `data-model-id` still holds the full ID so click-to-detail is unchanged
- Singletons rendered as before, but org prefix stripped

**CSS additions (`web/style.css`)**
- `.model-group-header td` — gray-100 background, monospace 12px bold
- `.model-subgroup-header td` — gray-50 background, monospace 11px bold, 1.5rem indent
- `.model-variant-row td.model-id-cell` — 1.5rem left padding
- `.model-variant-row.model-subgroup-child td.model-id-cell` — 2.75rem left padding

---

## 13 April 2026 (continued, session 5)

### UI Refinements: Companies Tab + Analysis/Historical Merge

**Companies tab — license distribution table**
- Replaced the Chart.js horizontal bar chart in the company detail view with a plain three-column table: License / Count / %. Percentage computed client-side from `model_count`. Added a `<thead>` with column headers.
- Removed `renderLicenseChart()`, `activeChart`, and the associated chart destroy guard entirely.

**Companies tab — models sub-panel**
- Added a "Models" section beneath the license distribution in the company detail panel.
- Fetches `GET /api/models?company_id={id}&limit=50&offset=0` asynchronously after the company detail renders.
- Renders a compact table (Model / License / Modality / Params / Released) using the existing `.models-table` CSS class. Shows "Showing 50 of N" note for companies with more than 50 models.
- Each row is clickable: clicking a model opens a full model detail view inline within the company detail panel, using the existing `renderModelDetail()` function. A "← Back" button restores the company view.

**Analysis + Historical — merged into one tab with sub-tabs**
- Removed the standalone "Historical" top-nav link and `#section-historical` element.
- `#section-analysis` now contains a sub-tab bar with two tabs: **Snapshot** (original analysis charts) and **Historical** (original historical charts). Both sub-panels use the existing `.analysis-container` / `.analysis-nav` / `.analysis-content` / `.analysis-notes` structure — no content was changed.
- `#historical` hash now redirects to `#analysis` and activates the Historical sub-tab for backwards compatibility.
- Replaced the debug-logged `setupAnalysis` / `setupHistorical` pair (and duplicate `hashchange` listeners) with a single `setupAnalysisSection()` coordinator and `switchAnalysisSubtab()` toggle. Removed all leftover `console.log` debug calls from the section router.
- Removed the temporary `border: 2px solid red` debug style from `#section-analysis`.

---

## 13 April 2026 (continued, session 4)

### Models Browser Tab

**Context**
The `#models` nav link and section container existed as a "Coming soon" placeholder. Goal: replace it with a fully functional filterable models browser backed by a new API router.

**API (`api/routers/models.py`)**
Three new endpoints registered in `api/app.py`:
- `GET /api/models` — paginated (50/page) filtered list. Query params: `company_id`, `license_slug`, `country_hq`, `data_source`, `modality`, `limit`, `offset`. Returns `{ total, models: [...] }` with company name and license slug joined in. Ordered by `release_date DESC NULLS LAST`.
- `GET /api/models/filters` — distinct dropdown values: countries (from `companies.country_hq`) and modalities (from `models.metadata->>'modality'`). Registered before the path route so FastAPI matches it ahead of `/{model_id:path}`.
- `GET /api/models/{model_id:path}` — full detail for one model. Uses `:path` type to capture slashes in HuggingFace-style model IDs (e.g. `01-ai/Yi-1.5-34B`). Returns all model columns plus joined company and license info; `metadata` JSONB serialized as a nested object.

**Frontend layout**
Two-panel section (`flex-direction: row`) — filter sidebar (220px) on the left, main panel (`flex: 1`) on the right. The main panel hosts two mutually exclusive views toggled by row click / back button:
- **Table view**: sticky-header table with 8 columns — Model, Company, License, Modality, Params (B), Released, Downloads, Likes. Paginated with Prev / Next buttons and a "Page X of Y · N models" counter.
- **Detail view**: back button in a fixed header bar; scrollable content below with stat cards (Company, Country, License, Source, Parameters, Modality, Released, Last Modified, Downloads, Likes), pipeline/framework/model-type fields, architecture pills, and a tag pill cloud (capped at 20 tags). License slug links through to `#licenses`.

**Filter sidebar**
Five `<select>` dropdowns populated on first activation (lazy load guard via `modelsLoaded` flag):
- Company — from `GET /api/companies`
- License — from `GET /api/licenses`
- Country — from `GET /api/models/filters`
- Source — static (huggingface / openai / anthropic / google)
- Modality — from `GET /api/models/filters`

Clear filters button resets all selects and reloads page 0. Result count displayed below filters.

**CSS additions (`web/style.css`)**
New classes: `.models-filter-panel`, `.models-filters`, `.filter-group`, `.filter-label`, `.filter-select`, `.clear-filters-btn`, `.models-result-count`, `.models-main-panel`, `.models-table`, `.models-table-wrap`, `.model-id-cell`, `.num-cell`, `.models-license-tag`, `.models-pagination`, `.models-pagination-info`, `.models-detail-header`, `.models-back-btn`, `.models-detail-content`, `.model-tags`, `.model-tag`, `.model-meta-text`.

**Verified**
- `GET /api/models/filters` → 9 countries, 5 modalities
- `GET /api/models` → 4,258 total
- `GET /api/models?country_hq=USA&modality=text` → 1,018 results
- `GET /api/models/01-ai/Yi-1.5-34B` → full detail with metadata, company, license

---

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
