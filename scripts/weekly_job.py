#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weekly model collection + digest generation.

Runs every Monday via GitHub Actions (.github/workflows/weekly-collect.yml).
Can also be triggered manually:
    python scripts/weekly_job.py

Requires env vars:
    DATABASE_URL  — Neon/PostgreSQL connection string
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

# Force UTF-8 output so Unicode characters don't crash on Windows cp1252 consoles
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from datetime import date, timedelta
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Load .env for local dev (two levels up from scripts/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)


# ── DB connection ────────────────────────────────────────────────────────────

def get_conn() -> psycopg2.extensions.connection:
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url)
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ.get("PGPORT", 5432)),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ.get("PGPASSWORD", ""),
    )


# ── Step 1: Run incremental collection ──────────────────────────────────────

def run_collection() -> None:
    print("=" * 60)
    print("STEP 1: Running incremental HuggingFace collection")
    print("=" * 60)
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(repo_root / "huggingface" / "collect.py")],
        cwd=str(repo_root),
        env={**os.environ},  # inherit DATABASE_URL etc.
    )
    if result.returncode != 0:
        print(f"WARNING: collect.py exited with code {result.returncode}")


# ── Step 2: Query new models from the last 7 days ───────────────────────────

def fetch_new_models(conn: psycopg2.extensions.connection) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                m.model_id,
                m.display_name,
                m.release_date,
                m.metadata,
                c.display_name  AS company_name,
                c.hf_handle     AS company_handle,
                l.slug          AS license_slug
            FROM models m
            LEFT JOIN companies c ON c.id = m.company_id
            LEFT JOIN licenses  l ON l.id = m.license_id
            WHERE m.release_date >= CURRENT_DATE - INTERVAL '7 days'
              AND m.data_source = 'huggingface'
            ORDER BY m.release_date DESC
            """
        )
        return [dict(r) for r in cur.fetchall()]


# ── Step 3: Compute stats ────────────────────────────────────────────────────

def compute_stats(models: list[dict]) -> dict:
    from collections import Counter

    by_company: Counter = Counter()
    by_license: Counter = Counter()
    by_modality: Counter = Counter()
    company_handles: dict[str, str] = {}

    for m in models:
        name = m["company_name"] or "Unknown"
        by_company[name] += 1
        company_handles[name] = m.get("company_handle") or ""

        slug = m["license_slug"] or "unknown"
        by_license[slug] += 1

        modality = (m.get("metadata") or {}).get("modality") or "unknown"
        by_modality[modality] += 1

    return {
        "by_company": [
            {"name": n, "hf_handle": company_handles.get(n, ""), "count": c}
            for n, c in by_company.most_common(10)
        ],
        "by_license": [
            {"slug": s, "count": c}
            for s, c in by_license.most_common(10)
        ],
        "by_modality": [
            {"modality": md, "count": c}
            for md, c in by_modality.most_common()
        ],
    }


# ── Step 4: Upsert digest into DB ────────────────────────────────────────────

def save_digest(
    conn: psycopg2.extensions.connection,
    week_start: date,
    week_end: date,
    new_models: int,
    stats: dict,
    narrative: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO weekly_digests
                (week_start, week_end, new_models, by_company, by_license, by_modality, narrative)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (week_start) DO UPDATE SET
                week_end    = EXCLUDED.week_end,
                new_models  = EXCLUDED.new_models,
                by_company  = EXCLUDED.by_company,
                by_license  = EXCLUDED.by_license,
                by_modality = EXCLUDED.by_modality,
                narrative   = EXCLUDED.narrative,
                generated_at = NOW()
            """,
            (
                week_start,
                week_end,
                new_models,
                json.dumps(stats["by_company"]),
                json.dumps(stats["by_license"]),
                json.dumps(stats["by_modality"]),
                narrative,
            ),
        )
    conn.commit()
    print(f"Digest saved for week {week_start} → {week_end} ({new_models} new models)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # Step 1: collect new models
    run_collection()

    # Determine week window (Mon–Sun of the current week)
    today = date.today()
    week_start = today - timedelta(days=today.weekday())   # Monday
    week_end   = week_start + timedelta(days=6)             # Sunday

    print()
    print("=" * 60)
    print(f"STEP 2: Generating digest for {week_start} -> {week_end}")
    print("=" * 60)

    conn = get_conn()
    try:
        # Step 2: query new models
        new_models = fetch_new_models(conn)
        print(f"  Found {len(new_models)} new models in the last 7 days")

        if len(new_models) == 0:
            print("  No new models — saving zero-count digest anyway")

        # Step 3: stats
        stats = compute_stats(new_models)

        # Step 4: save
        save_digest(conn, week_start, week_end, len(new_models), stats, "")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
