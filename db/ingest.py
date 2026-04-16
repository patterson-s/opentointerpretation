#!/usr/bin/env python3
"""
opentointerpretation — PostgreSQL ingestion script
Usage:
    python db/ingest.py --sample   # insert first 5 records from each source
    python db/ingest.py            # full ingestion
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from datetime import date

load_dotenv(override=True)  # override=True required on Windows

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent

HF_FILE    = BASE / "huggingface" / "hf_6jan2026_meta.json"
OPENAI_FILE  = BASE / "closed" / "openai_gpt_version_history.json"
CLAUDE_FILE  = BASE / "closed" / "claude_version_history.json"
GEMINI_FILE  = BASE / "closed" / "gemini_version_history.json"

# ── NULL-safe helpers ───────────────────────────────────────────────────────

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
    """Parse an ISO 8601 string (e.g. '2023-09-27T16:25:27.000Z') to a date."""
    if not val:
        return None
    s = str(val).strip()
    # Take only the date portion
    date_part = s[:10]
    try:
        return date.fromisoformat(date_part)
    except ValueError:
        return None


# ── Connection ──────────────────────────────────────────────────────────────

def get_conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ.get("PGPORT", 5432)),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
    )


# ── Lookup helpers ──────────────────────────────────────────────────────────

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


# ── Ingestion functions ─────────────────────────────────────────────────────

def ingest_hf(
    cur: psycopg2.extensions.cursor,
    records: list[dict],
    company_cache: dict,
    license_cache: dict,
) -> tuple[int, int, list[str]]:
    inserted = skipped = 0
    failures: list[str] = []

    for i, rec in enumerate(records):
        model_id = safe_str(rec.get("model_id"))
        if not model_id:
            failures.append(f"[hf/{i}] missing model_id — skipped")
            skipped += 1
            continue
        try:
            company_id = get_or_create_company(
                cur,
                display_name=safe_str(rec.get("display_org")) or "",
                hf_handle=safe_str(rec.get("handle")),
                country_hq=safe_str(rec.get("country_hq")),
                cache=company_cache,
            )
            lic_slug = safe_str(rec.get("license"))
            license_id = get_or_create_license(cur, lic_slug, license_cache) if lic_slug else None

            # Build metadata JSONB from enriched fields (if present)
            meta_keys = (
                "pipeline_tag", "modality", "num_parameters", "last_modified",
                "library_name", "gated", "language", "architectures",
                "model_type", "tags",
                # Extended API fields
                "base_model", "fine_tuned_from", "datasets", "eval_metrics",
                "trending_score", "inference", "vocab_size", "hidden_size",
                "num_hidden_layers", "num_attention_heads", "max_position_embeddings",
                "safetensors_total", "has_license_file", "has_readme",
                "disabled", "private",
            )
            metadata = {k: rec[k] for k in meta_keys if k in rec and rec[k] is not None}

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
            inserted += 1
            if (i + 1) % 100 == 0:
                print(f"  [hf] processed {i + 1}/{len(records)}")
        except Exception as e:
            failures.append(f"[hf/{i}] {model_id}: {e}")
            skipped += 1
            continue

    return inserted, skipped, failures


def ingest_closed(
    cur: psycopg2.extensions.cursor,
    records: list[dict],
    source: str,          # 'openai' | 'anthropic' | 'google'
    display_name: str,    # company display name to look up/create
    country_hq: str,
    company_cache: dict,
    license_cache: dict,
) -> tuple[int, int, list[str]]:
    inserted = skipped = 0
    failures: list[str] = []

    company_id = get_or_create_company(
        cur,
        display_name=display_name,
        hf_handle=None,
        country_hq=country_hq,
        cache=company_cache,
    )

    for i, rec in enumerate(records):
        # Use api_id for Anthropic if present, else model_id, else variant
        model_id = (
            safe_str(rec.get("api_id"))
            or safe_str(rec.get("model_id"))
            or safe_str(rec.get("variant"))
        )
        if not model_id:
            failures.append(f"[{source}/{i}] missing identifier — skipped")
            skipped += 1
            continue
        try:
            rd = safe_str(rec.get("release_date"))
            cur.execute(
                """
                INSERT INTO models
                    (company_id, license_id, model_id, display_name, data_source,
                     generation, api_id, release_date, context_tokens, notes)
                VALUES (%s, NULL, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (model_id) DO NOTHING
                """,
                (
                    company_id,
                    model_id,
                    safe_str(rec.get("variant")),
                    source,
                    safe_str(rec.get("generation")),
                    safe_str(rec.get("api_id")),
                    rd,
                    safe_int(rec.get("context_tokens")),
                    safe_str(rec.get("notes")),
                ),
            )
            inserted += 1
        except Exception as e:
            failures.append(f"[{source}/{i}] {model_id}: {e}")
            skipped += 1
            continue

    return inserted, skipped, failures


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest opentointerpretation data into PostgreSQL")
    parser.add_argument("--sample", action="store_true", help="Insert first 5 records per source only")
    args = parser.parse_args()

    limit = 5 if args.sample else None
    mode = "SAMPLE (first 5 per source)" if args.sample else "FULL"
    print(f"Mode: {mode}\n")

    # Load source files
    def load(path: Path) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data[:limit] if limit else data

    hf_data     = load(HF_FILE)
    openai_data = load(OPENAI_FILE)
    claude_data = load(CLAUDE_FILE)
    gemini_data = load(GEMINI_FILE)

    conn = get_conn()
    conn.autocommit = False
    cur = conn.cursor()

    company_cache: dict[str, int] = {}
    license_cache: dict[str, int] = {}

    total_inserted = total_skipped = 0
    all_failures: list[str] = []

    # 1. HuggingFace models
    print(f"Ingesting HuggingFace ({len(hf_data)} records)...")
    ins, sk, fail = ingest_hf(cur, hf_data, company_cache, license_cache)
    print(f"  inserted={ins}, skipped={sk}, failures={len(fail)}")
    total_inserted += ins; total_skipped += sk; all_failures.extend(fail)

    # 2. Closed-source models
    closed_sources = [
        (openai_data, "openai",    "OpenAI",          "USA"),
        (claude_data, "anthropic", "Anthropic",        "USA"),
        (gemini_data, "google",    "Google DeepMind",  "USA"),
    ]
    for data, src, dname, country in closed_sources:
        print(f"Ingesting {dname} ({len(data)} records)...")
        ins, sk, fail = ingest_closed(cur, data, src, dname, country, company_cache, license_cache)
        print(f"  inserted={ins}, skipped={sk}, failures={len(fail)}")
        total_inserted += ins; total_skipped += sk; all_failures.extend(fail)

    conn.commit()

    print(f"\nIngestion complete: {total_inserted} inserted, {total_skipped} skipped, {len(all_failures)} failed")
    if all_failures:
        print("Failed records:")
        for f in all_failures:
            print(f"  {f}")

    # Sample output in --sample mode
    if args.sample:
        print("\n--- Sample rows ---")
        for table in ("companies", "licenses", "models"):
            cur.execute(f"SELECT * FROM {table} LIMIT 5")
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            print(f"\n{table}:")
            print("  " + " | ".join(cols))
            for row in rows:
                print("  " + " | ".join(str(v) for v in row))

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
