#!/usr/bin/env python3
"""
Incremental HuggingFace model collection.

Fetches all current models from the tracked org list, compares against the DB,
enriches new models (license + metadata), and inserts them directly into PostgreSQL.
Records each run in the collection_runs table.

Usage:
    python huggingface/collect.py              # full run
    python huggingface/collect.py --dry-run    # preview; no DB writes
    python huggingface/collect.py --limit 10   # enrich first 10 new models only
    python huggingface/collect.py --sample     # count new models; no enrichment
    python huggingface/collect.py --delay 0.5  # override 0.3s API delay
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import date
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Allow running as a script from the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(override=True)

from huggingface.hf_pipeline import (  # noqa: E402
    ORG_HANDLES,
    fetch_models,
    apply_manual_corrections,
    add_country,
)
from huggingface.enrich_hf_metadata import (  # noqa: E402
    fetch_hf_model_api,
    extract_metadata_fields,
)
from huggingface.enrich_hf_licenses import (  # noqa: E402
    resolve_license,
    resolve_other_license,
)


# ── NULL-safe helpers (mirrored from db/ingest.py) ─────────────────────────

def safe_str(val: Any, max_len: int | None = None) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    if s.lower() in ("", "unknown", "n/a", "none", "null", "-"):
        return None
    return s[:max_len] if max_len else s


def safe_int(val: Any) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def safe_date(val: Any) -> date | None:
    if not val:
        return None
    s = str(val).strip()
    date_part = s[:10]
    try:
        return date.fromisoformat(date_part)
    except ValueError:
        return None


# ── Connection ──────────────────────────────────────────────────────────────

def get_conn() -> psycopg2.extensions.connection:
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url)
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ.get("PGPORT", 5432)),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
    )


# ── Lookup helpers (mirrored from db/ingest.py) ─────────────────────────────

def get_or_create_company(
    cur: psycopg2.extensions.cursor,
    display_name: str,
    hf_handle: str | None,
    country_hq: str | None,
    cache: dict[str, int],
) -> int | None:
    if not display_name:
        return None
    if display_name in cache:
        return cache[display_name]
    cur.execute(
        """
        INSERT INTO companies (display_name, hf_handle, country_hq)
        VALUES (%s, %s, %s)
        ON CONFLICT (display_name) DO UPDATE
            SET hf_handle  = COALESCE(EXCLUDED.hf_handle,  companies.hf_handle),
                country_hq = COALESCE(EXCLUDED.country_hq, companies.country_hq)
        RETURNING id
        """,
        (display_name, safe_str(hf_handle), safe_str(country_hq)),
    )
    row = cur.fetchone()
    cid = row[0]
    cache[display_name] = cid
    return cid


def get_or_create_license(
    cur: psycopg2.extensions.cursor,
    slug: str,
    cache: dict[str, int],
) -> int | None:
    if not slug:
        return None
    if slug in cache:
        return cache[slug]
    cur.execute(
        """
        INSERT INTO licenses (slug)
        VALUES (%s)
        ON CONFLICT (slug) DO NOTHING
        RETURNING id
        """,
        (slug,),
    )
    row = cur.fetchone()
    if row:
        lid = row[0]
    else:
        cur.execute("SELECT id FROM licenses WHERE slug = %s", (slug,))
        lid = cur.fetchone()[0]
    cache[slug] = lid
    return lid


# ── Collection run tracking ──────────────────────────────────────────────────

def start_run(cur: psycopg2.extensions.cursor) -> int:
    cur.execute(
        "INSERT INTO collection_runs (status) VALUES ('running') RETURNING id"
    )
    return cur.fetchone()[0]


def finish_run(
    cur: psycopg2.extensions.cursor,
    run_id: int,
    status: str,
    total_seen: int | None = None,
    added: int | None = None,
    error: str | None = None,
) -> None:
    cur.execute(
        """
        UPDATE collection_runs
        SET status = %s,
            completed_at = NOW(),
            total_models_seen = %s,
            new_models_added = %s,
            error_message = %s
        WHERE id = %s
        """,
        (status, total_seen, added, error, run_id),
    )


# ── DB query helpers ─────────────────────────────────────────────────────────

def get_existing_hf_model_ids(cur: psycopg2.extensions.cursor) -> set[str]:
    cur.execute("SELECT model_id FROM models WHERE data_source = 'huggingface'")
    return {row[0] for row in cur.fetchall()}


# ── Model insertion ──────────────────────────────────────────────────────────

META_KEYS = (
    "pipeline_tag", "modality", "num_parameters", "last_modified",
    "library_name", "gated", "language", "architectures",
    "model_type", "tags",
    "base_model", "fine_tuned_from", "datasets", "eval_metrics",
    "trending_score", "inference", "vocab_size", "hidden_size",
    "num_hidden_layers", "num_attention_heads", "max_position_embeddings",
    "safetensors_total", "has_license_file", "has_readme",
    "disabled", "private",
)


def insert_hf_model(
    cur: psycopg2.extensions.cursor,
    rec: dict,
    company_cache: dict[str, int],
    license_cache: dict[str, int],
) -> bool:
    """Insert one model; returns True if actually inserted (not a conflict skip)."""
    model_id = safe_str(rec.get("model_id"))
    if not model_id:
        return False

    company_id = get_or_create_company(
        cur,
        display_name=safe_str(rec.get("display_org")) or "",
        hf_handle=safe_str(rec.get("handle")),
        country_hq=safe_str(rec.get("country_hq")),
        cache=company_cache,
    )
    lic_slug = safe_str(rec.get("license"))
    license_id = get_or_create_license(cur, lic_slug, license_cache) if lic_slug else None

    metadata = {k: rec[k] for k in META_KEYS if k in rec and rec[k] is not None}

    cur.execute(
        """
        INSERT INTO models
            (company_id, license_id, model_id, url, data_source,
             likes, downloads, release_date, metadata)
        VALUES (%s, %s, %s, %s, 'huggingface', %s, %s, %s, %s)
        ON CONFLICT (model_id) DO NOTHING
        """,
        (
            company_id,
            license_id,
            model_id,
            safe_str(rec.get("url")),
            safe_int(rec.get("likes")),
            safe_int(rec.get("downloads")),
            safe_date(rec.get("release_date")),
            psycopg2.extras.Json(metadata) if metadata else None,
        ),
    )
    return cur.rowcount > 0


# ── License enrichment ────────────────────────────────────────────────────────

def enrich_license(rec: dict) -> dict:
    """Resolve license for a record using the full 5-layer chain."""
    repo_id = rec.get("model_id", "")
    repo_url = rec.get("url") or f"https://huggingface.co/{repo_id}"
    current = (rec.get("license") or "").strip()

    if not current:
        outcome = resolve_license(repo_id, repo_url)
        if outcome.license:
            rec["license"] = outcome.license
    elif current == "other":
        other = resolve_other_license(repo_id, repo_url)
        if other.license_slug:
            rec["license"] = other.license_slug

    return rec


# ── Main collection logic ─────────────────────────────────────────────────────

def run_collection(
    dry_run: bool,
    limit: int | None,
    delay: float,
    sample: bool,
) -> None:
    conn = get_conn()
    conn.autocommit = False
    cur = conn.cursor()

    run_id: int | None = None
    if not dry_run:
        run_id = start_run(cur)
        conn.commit()
        print(f"Started collection run #{run_id}")

    try:
        # Phase 1: fetch all current HF models
        print("Fetching models from HuggingFace...")
        hf_models: dict[str, dict] = {}
        for display_org, handles in ORG_HANDLES.items():
            records = fetch_models(display_org, handles)
            for r in records:
                hf_models[r["model_id"]] = r
            print(f"  {display_org}: {len(records)} models")

        total_seen = len(hf_models)
        print(f"\nTotal models on HF: {total_seen}")

        # Phase 2: diff against DB
        print("Querying DB for existing model IDs...")
        existing_ids = get_existing_hf_model_ids(cur)
        print(f"Existing in DB: {len(existing_ids)}")

        new_ids = set(hf_models.keys()) - existing_ids
        new_records = [hf_models[mid] for mid in sorted(new_ids)]
        print(f"New models to add: {len(new_records)}")

        # Sample mode: report counts only
        if sample:
            print("\n[--sample] Skipping enrichment and insertion.")
            if not dry_run and run_id is not None:
                finish_run(cur, run_id, "completed", total_seen, 0)
                conn.commit()
            return

        if not new_records:
            print("Nothing to do.")
            if not dry_run and run_id is not None:
                finish_run(cur, run_id, "completed", total_seen, 0)
                conn.commit()
            return

        if limit:
            new_records = new_records[:limit]
            print(f"Applying --limit {limit}: processing {len(new_records)} models")

        # Phase 3: enrich and insert
        company_cache: dict[str, int] = {}
        license_cache: dict[str, int] = {}
        added = 0
        BATCH = 50

        for i, rec in enumerate(new_records, start=1):
            model_id = rec.get("model_id", "?")
            print(f"  [{i}/{len(new_records)}] {model_id}", end="", flush=True)

            try:
                # Apply manual license overrides and country
                rec = apply_manual_corrections(rec)
                rec = add_country(rec)

                # License enrichment (full 5-layer chain)
                rec = enrich_license(rec)

                # Metadata enrichment
                time.sleep(delay)
                api_data = fetch_hf_model_api(model_id)
                if api_data:
                    meta_fields = extract_metadata_fields(api_data)
                    rec.update(meta_fields)

                if not dry_run:
                    did_insert = insert_hf_model(cur, rec, company_cache, license_cache)
                    if did_insert:
                        added += 1
                        print(f" ✓ license={rec.get('license', '-')}", flush=True)
                    else:
                        print(f" (conflict skip)", flush=True)
                else:
                    print(f" [dry-run] license={rec.get('license', '-')}", flush=True)

                # Batch commit
                if not dry_run and i % BATCH == 0:
                    conn.commit()
                    print(f"  -- committed batch ({i} processed, {added} added so far)")

            except Exception as e:
                print(f" ERROR: {e}", flush=True)
                continue

        # Final commit
        if not dry_run:
            conn.commit()
            print(f"\nDone. Added {added} new models.")
            if run_id is not None:
                finish_run(cur, run_id, "completed", total_seen, added)
                conn.commit()
        else:
            print(f"\n[dry-run] Would add up to {len(new_records)} new models.")

    except (KeyboardInterrupt, Exception) as exc:
        err_msg = traceback.format_exc()
        print(f"\nInterrupted: {exc}")
        if not dry_run and run_id is not None:
            try:
                finish_run(cur, run_id, "failed", error=err_msg)
                conn.commit()
            except Exception:
                pass
        raise

    finally:
        cur.close()
        conn.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incremental HuggingFace model collection"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview new models; do not write to DB",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N new models (for testing)",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Count new models only; skip enrichment and insertion",
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="Seconds between HF API calls (default: 0.3)",
    )
    args = parser.parse_args()

    run_collection(
        dry_run=args.dry_run,
        limit=args.limit,
        delay=args.delay,
        sample=args.sample,
    )


if __name__ == "__main__":
    main()
