#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 3: Classify all models using the approved typology scheme via Cohere Command A.

Runs parallel classification with a thread pool (default 8 workers) so 4,258 models
finish in minutes rather than hours.

Results are written to:
  - models.metadata JSONB (typology_type, typology_tags)
  - analysis/typology/typology_results.json  (full audit trail)
  - analysis/typology/classify.ckpt.json     (resume checkpoint)

Usage:
    python analysis/typology/03_classify_models.py
    python analysis/typology/03_classify_models.py --dry-run       # no DB writes
    python analysis/typology/03_classify_models.py --workers 12    # more parallelism
    python analysis/typology/03_classify_models.py --limit 50      # test first N models
    python analysis/typology/03_classify_models.py --company "Mistral AI"
    python analysis/typology/03_classify_models.py --reset         # clear checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cohere
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "analysis" / "typology"
SCHEME_FILE = OUT_DIR / "typology_scheme.json"
RESULTS_FILE = OUT_DIR / "typology_results.json"
CKPT_FILE = OUT_DIR / "classify.ckpt.json"

load_dotenv(ROOT / ".env", override=True)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ.get("PGPORT", 5432)),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ.get("PGPASSWORD", ""),
    )


def fetch_all_models(conn, company_filter: str | None = None) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if company_filter:
            cur.execute(
                """
                SELECT
                    m.model_id,
                    m.display_name,
                    (m.metadata->>'num_parameters')::float AS num_parameters,
                    m.data_source,
                    c.display_name                AS company_name,
                    m.metadata->>'pipeline_tag'   AS pipeline_tag,
                    m.metadata->>'modality'       AS modality,
                    m.metadata->>'model_type'     AS model_type,
                    m.metadata->'tags'            AS tags
                FROM models m
                JOIN companies c ON m.company_id = c.id
                WHERE c.display_name = %s
                ORDER BY m.model_id
                """,
                (company_filter,),
            )
        else:
            cur.execute(
                """
                SELECT
                    m.model_id,
                    m.display_name,
                    (m.metadata->>'num_parameters')::float AS num_parameters,
                    m.data_source,
                    c.display_name                AS company_name,
                    m.metadata->>'pipeline_tag'   AS pipeline_tag,
                    m.metadata->>'modality'       AS modality,
                    m.metadata->>'model_type'     AS model_type,
                    m.metadata->'tags'            AS tags
                FROM models m
                JOIN companies c ON m.company_id = c.id
                ORDER BY c.display_name, m.model_id
                """
            )
        return [dict(r) for r in cur.fetchall()]


# Thread-local DB connections for parallel writes
_local = threading.local()

def _get_thread_conn():
    if not hasattr(_local, "conn") or _local.conn.closed:
        _local.conn = get_conn()
    return _local.conn


def write_typology_to_db(model_id: str, typ: str, tags: list[str]) -> None:
    patch = json.dumps({"typology_type": typ, "typology_tags": tags})
    conn = _get_thread_conn()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE models SET metadata = metadata || %s::jsonb WHERE model_id = %s",
            (patch, model_id),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Cohere classification
# ---------------------------------------------------------------------------

def build_classification_prompt(scheme: dict, model: dict) -> str:
    type_defs = "\n".join(
        f"  - {t['id']}: {t['definition']}"
        for t in scheme["types"]
    )
    tag_list = ", ".join(t["id"] for t in scheme["tags"])

    params = (
        f"{model['num_parameters']:.1f}B"
        if model.get("num_parameters")
        else "unknown"
    )
    tags_raw = model.get("tags") or []
    if isinstance(tags_raw, str):
        try:
            tags_raw = json.loads(tags_raw)
        except Exception:
            tags_raw = []
    hf_tags = ", ".join(tags_raw[:12]) if tags_raw else "none"

    return (
        f"Classify this AI model into exactly one type and assign relevant tags.\n\n"
        f"MODEL:\n"
        f"  id: {model['model_id']}\n"
        f"  company: {model['company_name']}\n"
        f"  pipeline_tag: {model.get('pipeline_tag') or 'unknown'}\n"
        f"  modality: {model.get('modality') or 'unknown'}\n"
        f"  model_type: {model.get('model_type') or 'unknown'}\n"
        f"  parameters: {params}\n"
        f"  hf_tags: {hf_tags}\n\n"
        f"TYPOLOGY TYPES (choose exactly one):\n{type_defs}\n\n"
        f"AVAILABLE TAGS (choose all that apply, from this list): {tag_list}\n\n"
        f"Return JSON only:\n"
        f'  {{"type": "<type_id>", "tags": ["tag1", "tag2"], "confidence": "high|medium|low", "rationale": "<one sentence>"}}'
    )


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["type", "tags"],
}


def classify_model(co: cohere.ClientV2, scheme: dict, model: dict) -> dict:
    prompt = build_classification_prompt(scheme, model)
    response = co.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_object",
            "json_schema": RESPONSE_SCHEMA,
        },
        temperature=0.0,
    )
    raw = response.message.content[0].text
    result = json.loads(raw)

    valid_types = {t["id"] for t in scheme["types"]}
    if result.get("type") not in valid_types:
        result["type"] = "accessory"
        result["confidence"] = "low"

    valid_tags = {t["id"] for t in scheme["tags"]}
    result["tags"] = [t for t in result.get("tags", []) if t in valid_tags]

    return result


# ---------------------------------------------------------------------------
# Checkpoint helpers (thread-safe)
# ---------------------------------------------------------------------------

_ckpt_lock = threading.Lock()
_results: dict = {}


def load_checkpoint() -> dict:
    if CKPT_FILE.exists():
        return json.loads(CKPT_FILE.read_text())
    return {}


def save_checkpoint() -> None:
    with _ckpt_lock:
        CKPT_FILE.write_text(json.dumps(_results, indent=2))


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def process_model(
    model: dict,
    scheme: dict,
    dry_run: bool,
    co: cohere.ClientV2,
) -> tuple[str, dict | None, str | None]:
    """Returns (model_id, result_dict, error_str)."""
    mid = model["model_id"]
    try:
        classification = classify_model(co, scheme, model)
        record = {
            "model_id": mid,
            "company": model["company_name"],
            "type": classification["type"],
            "tags": classification["tags"],
            "confidence": classification.get("confidence", "?"),
            "rationale": classification.get("rationale", ""),
        }
        if not dry_run:
            write_typology_to_db(mid, classification["type"], classification["tags"])
        return mid, record, None
    except Exception as exc:
        return mid, None, str(exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_stop_requested = False

def _handle_sigint(sig, frame):
    global _stop_requested
    if not _stop_requested:
        print("\n[Ctrl+C] Stopping after in-flight requests finish — saving progress...", flush=True)
        _stop_requested = True
    else:
        print("\n[Ctrl+C] Force exit.", flush=True)
        sys.exit(1)


def run(
    dry_run: bool = False,
    limit: int | None = None,
    company_filter: str | None = None,
    reset: bool = False,
    workers: int = 8,
) -> None:
    global _results, _stop_requested
    signal.signal(signal.SIGINT, _handle_sigint)

    if not SCHEME_FILE.exists():
        print(f"[ERROR] Scheme file not found: {SCHEME_FILE}")
        print("Run 02_synthesize_typology.py first.")
        sys.exit(1)

    scheme = json.loads(SCHEME_FILE.read_text())
    print(f"Loaded typology scheme: {len(scheme['types'])} types, {len(scheme['tags'])} tags")

    if reset and CKPT_FILE.exists():
        CKPT_FILE.unlink()
        print("Checkpoint cleared.")

    # Each worker gets its own Cohere client (not thread-safe to share)
    # We'll create one per thread via thread-local storage
    _co_local = threading.local()

    def get_co():
        if not hasattr(_co_local, "client"):
            _co_local.client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
        return _co_local.client

    conn = get_conn()
    models = fetch_all_models(conn, company_filter)
    conn.close()

    if limit:
        models = models[:limit]

    _results = load_checkpoint()
    pending = [m for m in models if m["model_id"] not in _results]

    total = len(models)
    already_done = total - len(pending)
    print(f"Total models: {total} | Already done: {already_done} | To classify: {len(pending)}")
    if not pending:
        print("All models already classified.")
    else:
        print(f"Running with {workers} parallel workers...")

    completed = already_done
    errors = 0
    last_save = time.time()

    def worker(model):
        co = get_co()
        return process_model(model, scheme, dry_run, co)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, m): m["model_id"] for m in pending}

        for future in as_completed(futures):
            if _stop_requested:
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                save_checkpoint()
                print(f"Stopped. Saved {completed}/{total} classifications.")
                break

            mid, record, error = future.result()
            if error:
                errors += 1
                _results[mid] = {"model_id": mid, "error": error}
                print(f"  [ERROR] {mid}: {error}", flush=True)
            else:
                completed += 1
                with _ckpt_lock:
                    _results[mid] = record

            # Save checkpoint every 100 completions or 30 seconds
            if completed % 100 == 0 or (time.time() - last_save) > 30:
                save_checkpoint()
                last_save = time.time()
                pct = completed / total * 100
                print(f"  [{completed}/{total} {pct:.0f}%] errors={errors}", flush=True)

    save_checkpoint()

    # Write final output
    output = list(_results.values())
    RESULTS_FILE.write_text(json.dumps(output, indent=2))

    classified = [r for r in output if "type" in r]
    type_counts = Counter(r["type"] for r in classified)
    tag_counts: Counter = Counter()
    for r in classified:
        tag_counts.update(r.get("tags", []))

    print(f"\n{'='*50}")
    print(f"CLASSIFICATION COMPLETE{' (DRY RUN)' if dry_run else ''}")
    print(f"{'='*50}")
    print(f"Total: {len(output)} | Classified: {len(classified)} | Errors: {errors}")
    print(f"\nType distribution:")
    for typ, count in type_counts.most_common():
        print(f"  {typ}: {count}")
    print(f"\nTop 15 tags:")
    for tag, count in tag_counts.most_common(15):
        print(f"  {tag}: {count}")
    print(f"\nResults -> {RESULTS_FILE}")
    if not dry_run:
        print("DB updated: models.metadata contains typology_type + typology_tags")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: classify all models (parallel)")
    parser.add_argument("--dry-run", action="store_true", help="No DB writes")
    parser.add_argument("--limit", type=int, help="Process only first N models")
    parser.add_argument("--company", help="Classify only models for one company")
    parser.add_argument("--reset", action="store_true", help="Clear checkpoint before running")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default 8)")
    args = parser.parse_args()
    run(
        dry_run=args.dry_run,
        limit=args.limit,
        company_filter=args.company,
        reset=args.reset,
        workers=args.workers,
    )
