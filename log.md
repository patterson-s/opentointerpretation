# Project Log

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
