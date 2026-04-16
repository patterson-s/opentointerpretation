#!/usr/bin/env python3
"""
Fetch raw README.md (model card) for every HuggingFace model and store it in
the `model_card_raw` column of the `models` table.

The DB itself is the checkpoint: rows with `model_card_fetched_at IS NULL` are
processed first; re-running skips already-fetched rows unless --force is used.

Non-200 responses (404, 403, network errors) are stored as NULL with the
timestamp set, so the script does not retry them on re-runs unless --force.
Misses are also logged to huggingface/fetch_model_cards.misses.txt.

Usage:
    python huggingface/fetch_model_cards.py               # fetch all missing
    python huggingface/fetch_model_cards.py --limit 10    # test first 10
    python huggingface/fetch_model_cards.py --force       # re-fetch all
    python huggingface/fetch_model_cards.py --model-id meta-llama/Llama-2-7b-hf
    python huggingface/fetch_model_cards.py --dry-run     # print counts only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

BASE = Path(__file__).resolve().parent.parent
MISSES_PATH = Path(__file__).resolve().parent / "fetch_model_cards.misses.txt"

README_URL = "https://huggingface.co/{model_id}/raw/main/README.md"
HTTP_TIMEOUT = 20
UA = {"User-Agent": "opentointerpretation-modelcard-bot/1.0"}
BATCH_SIZE = 50


# ── DB ──────────────────────────────────────────────────────────────────────

def get_conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ.get("PGPORT", 5432)),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
    )


def get_pending_model_ids(
    cur: psycopg2.extensions.cursor,
    force: bool,
    model_id_filter: Optional[str],
) -> list[str]:
    """Return model_ids to process, ordered by DB id for stable resume."""
    if model_id_filter:
        cur.execute(
            "SELECT model_id FROM models WHERE model_id = %s AND data_source = 'huggingface'",
            (model_id_filter,),
        )
    elif force:
        cur.execute(
            "SELECT model_id FROM models WHERE data_source = 'huggingface' ORDER BY id"
        )
    else:
        cur.execute(
            "SELECT model_id FROM models "
            "WHERE data_source = 'huggingface' AND model_card_fetched_at IS NULL "
            "ORDER BY id"
        )
    return [row[0] for row in cur.fetchall()]


# ── HTTP ─────────────────────────────────────────────────────────────────────

def fetch_readme(
    model_id: str, session: requests.Session
) -> tuple[Optional[str], str]:
    """
    Fetch README.md for model_id.
    Returns (text_or_None, status) where status is one of:
      "ok", "404", "403", "empty", "error:<code>", "exception"
    """
    url = README_URL.format(model_id=model_id)
    try:
        r = session.get(url, timeout=HTTP_TIMEOUT, headers=UA)
        if r.status_code == 200:
            text = r.text.strip()
            if not text:
                return None, "empty"
            return text, "ok"
        elif r.status_code == 404:
            return None, "404"
        elif r.status_code == 403:
            return None, "403"
        else:
            return None, f"error:{r.status_code}"
    except Exception as e:
        return None, f"exception:{type(e).__name__}"


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fetch HuggingFace README.md model cards and store in DB."
    )
    ap.add_argument(
        "--delay", type=float, default=0.35,
        help="Seconds between HTTP requests (default 0.35)"
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N models (for testing)"
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-fetch even if model_card_raw is already populated"
    )
    ap.add_argument(
        "--model-id", dest="model_id", default=None,
        help="Fetch only this specific model_id"
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print pending count without fetching"
    )
    args = ap.parse_args()

    conn = get_conn()
    conn.autocommit = False
    cur = conn.cursor()

    pending = get_pending_model_ids(cur, force=args.force, model_id_filter=args.model_id)
    if args.limit:
        pending = pending[: args.limit]

    print(f"Pending models: {len(pending)}"
          f"{' (force re-fetch)' if args.force else ''}")

    if args.dry_run or not pending:
        cur.close()
        conn.close()
        return

    eta_min = len(pending) * args.delay / 60
    print(f"Delay: {args.delay}s/call | ETA: ~{eta_min:.0f} min | "
          f"Batch commit every {BATCH_SIZE} rows")
    print(f"Misses log: {MISSES_PATH}")
    print()

    session = requests.Session()
    misses: list[str] = []

    fetched = ok = 0
    batch_updates: list[tuple] = []
    now = datetime.now(timezone.utc)

    update_sql = """
        UPDATE models
        SET model_card_raw = %s, model_card_fetched_at = %s
        WHERE model_id = %s
    """

    try:
        for i, model_id in enumerate(pending):
            text, status = fetch_readme(model_id, session)

            if status == "ok":
                ok += 1
            else:
                misses.append(f"{model_id}\t{status}")

            batch_updates.append((text, now, model_id))
            fetched += 1

            if args.delay > 0:
                time.sleep(args.delay)

            if len(batch_updates) >= BATCH_SIZE:
                cur.executemany(update_sql, batch_updates)
                conn.commit()
                batch_updates = []
                print(
                    f"  [{fetched}/{len(pending)}] committed batch "
                    f"(ok so far: {ok}, misses: {fetched - ok})"
                )

    except KeyboardInterrupt:
        print("\nInterrupted — flushing current batch ...")
        if batch_updates:
            cur.executemany(update_sql, batch_updates)
            conn.commit()
        print("Partial progress saved to DB.")
    else:
        if batch_updates:
            cur.executemany(update_sql, batch_updates)
            conn.commit()

    cur.close()
    conn.close()

    # Write misses log
    if misses:
        with open(MISSES_PATH, "a", encoding="utf-8") as f:
            f.write("\n".join(misses) + "\n")

    print(f"\nDone. fetched={fetched}, ok={ok}, misses={fetched - ok}")
    if misses:
        print(f"Miss log written to: {MISSES_PATH}")


if __name__ == "__main__":
    main()
