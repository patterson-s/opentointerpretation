#!/usr/bin/env python3
"""
Merge new metadata fields into existing models rows using PostgreSQL's JSONB
merge operator (||). Safe to re-run: existing keys are not overwritten unless
--overwrite is passed.

Usage:
    python db/update_metadata.py --input huggingface/hf_6jan2026_meta_v2.json
    python db/update_metadata.py --input huggingface/hf_6jan2026_meta_v2.json --dry-run
    python db/update_metadata.py --input huggingface/hf_6jan2026_meta_v2.json --overwrite
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

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

BASE = Path(__file__).resolve().parent.parent

# Keys to extract from the enriched JSON and merge into metadata JSONB.
# These are the fields added in the v2 enrichment pass.
NEW_META_KEYS = (
    "base_model", "fine_tuned_from", "datasets", "eval_metrics",
    "trending_score", "inference", "vocab_size", "hidden_size",
    "num_hidden_layers", "num_attention_heads", "max_position_embeddings",
    "safetensors_total", "has_license_file", "has_readme",
    "disabled", "private",
)

BATCH_SIZE = 50


def get_conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ.get("PGPORT", 5432)),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
    )


def load_records(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_patch(rec: dict[str, Any]) -> dict[str, Any]:
    """Extract only the new metadata keys from a record."""
    return {k: rec[k] for k in NEW_META_KEYS if k in rec and rec[k] is not None}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge new metadata fields into existing models rows."
    )
    ap.add_argument("--input", required=True, help="Path to enriched HF JSON array")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be updated without writing to DB"
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Use jsonb_build_object merge that overwrites existing keys (default: only adds new keys)"
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N records (for testing)"
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    records = load_records(in_path)
    if args.limit:
        records = records[: args.limit]

    print(f"Loaded {len(records)} records from: {in_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'WRITE'} | "
          f"Merge strategy: {'overwrite' if args.overwrite else 'additive (no overwrite)'}")
    print(f"New keys to merge: {', '.join(NEW_META_KEYS)}")
    print()

    if args.dry_run:
        # Sample: show first 5 patches
        shown = 0
        for rec in records:
            mid = rec.get("model_id")
            patch = build_patch(rec)
            if patch and shown < 5:
                print(f"  {mid}: {list(patch.keys())}")
                shown += 1
        total_with_patch = sum(1 for r in records if build_patch(r))
        print(f"\n{total_with_patch}/{len(records)} records have new metadata fields.")
        return

    conn = get_conn()
    conn.autocommit = False
    cur = conn.cursor()

    if args.overwrite:
        # metadata || patch — patch keys overwrite existing values
        sql = "UPDATE models SET metadata = metadata || %s::jsonb WHERE model_id = %s"
    else:
        # patch || metadata — existing keys take precedence; only new keys are added
        sql = "UPDATE models SET metadata = %s::jsonb || metadata WHERE model_id = %s"

    updated = skipped = no_patch = 0
    batch: list[tuple[str, str]] = []

    for i, rec in enumerate(records):
        model_id = rec.get("model_id")
        if not model_id:
            skipped += 1
            continue

        patch = build_patch(rec)
        if not patch:
            no_patch += 1
            continue

        batch.append((json.dumps(patch), model_id))

        if len(batch) >= BATCH_SIZE:
            cur.executemany(sql, batch)
            conn.commit()
            updated += len(batch)
            batch = []
            print(f"  committed {updated}/{len(records)} ...")

    if batch:
        cur.executemany(sql, batch)
        conn.commit()
        updated += len(batch)

    cur.close()
    conn.close()

    print(f"\nDone. updated={updated}, no_patch={no_patch}, skipped={skipped}")


if __name__ == "__main__":
    main()
