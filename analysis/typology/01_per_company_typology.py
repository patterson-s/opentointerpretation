#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1: Per-company typology generation using Claude Sonnet.

For each company, sends all model metadata to claude-sonnet-4-6 and asks it
to describe the typological patterns observed — what types of models the company
releases and what distinguishing characteristics (tags) each cluster has.

Output: analysis/typology/company_typologies.json
        analysis/typology/company_typologies.ckpt.json  (resume checkpoint)

Usage:
    python analysis/typology/01_per_company_typology.py
    python analysis/typology/01_per_company_typology.py --company "Mistral AI"
    python analysis/typology/01_per_company_typology.py --limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import anthropic
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "analysis" / "typology"
OUT_FILE = OUT_DIR / "company_typologies.json"
CKPT_FILE = OUT_DIR / "company_typologies.ckpt.json"

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


def fetch_companies(conn) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, display_name FROM companies ORDER BY display_name"
        )
        return [dict(r) for r in cur.fetchall()]


def fetch_models_for_company(conn, company_id: int) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                m.model_id,
                m.display_name,
                (m.metadata->>'num_parameters')::float AS num_parameters,
                m.data_source,
                m.metadata->>'pipeline_tag'  AS pipeline_tag,
                m.metadata->>'modality'      AS modality,
                m.metadata->>'model_type'    AS model_type,
                m.metadata->'tags'           AS tags
            FROM models m
            WHERE m.company_id = %s
            ORDER BY m.model_id
            """,
            (company_id,),
        )
        return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Claude helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI industry analyst specializing in model taxonomy.
Your job is to examine a company's model portfolio and characterize what types of
models they release, using a consistent typology framework.

The typology uses:
- **types** (mutually exclusive): general-llm | specialized-llm | accessory
  - general-llm: broad-capability language/chat/completion model
  - specialized-llm: LM with a primary domain/capability focus (code, vision, reasoning, etc.)
  - accessory: non-generation model supporting ML pipelines (embeddings, rerankers,
    classifiers, ASR, image encoders, segmentation, depth estimation, etc.)
- **tags** (multiple, overlapping): descriptive labels for the model's characteristics
  Examples: base, instruction-tuned, rlhf, code, math, reasoning, vision, audio,
  multimodal, multilingual, medical, legal, embedding, reranker, classifier,
  quantized, distilled, chat, long-context, agent, tool-use, safety

Return your analysis as a JSON object with this schema:
{
  "company": "<name>",
  "summary": "<1-2 sentence overview of the company's model strategy>",
  "observed_types": {
    "general-llm": <integer count>,
    "specialized-llm": <integer count>,
    "accessory": <integer count>
  },
  "clusters": [
    {
      "label": "<cluster name, e.g. 'Command series', 'Embedding models'>",
      "type": "general-llm|specialized-llm|accessory",
      "suggested_tags": ["tag1", "tag2"],
      "model_ids": ["model_id1", "model_id2"],
      "notes": "<optional explanation>"
    }
  ],
  "suggested_tags": ["overall list of tags relevant to this company's portfolio"]
}"""


def models_to_text(models: list[dict]) -> str:
    """Convert model list to a compact text representation for the prompt."""
    lines = []
    for m in models:
        params = f"{m['num_parameters']:.1f}B" if m.get("num_parameters") else "?"
        tags_raw = m.get("tags") or []
        if isinstance(tags_raw, str):
            try:
                tags_raw = json.loads(tags_raw)
            except Exception:
                tags_raw = []
        tags_str = ", ".join(tags_raw[:8]) if tags_raw else ""
        lines.append(
            f"- {m['model_id']} | {m.get('pipeline_tag') or '?'} | "
            f"{m.get('modality') or '?'} | {m.get('model_type') or '?'} | "
            f"{params} | [{tags_str}]"
        )
    return "\n".join(lines)


def analyze_company_batch(
    client: anthropic.Anthropic,
    company_name: str,
    models: list[dict],
    batch_idx: int = 0,
    total_batches: int = 1,
) -> dict:
    """Send a batch of models to Claude and return the parsed JSON response."""
    model_text = models_to_text(models)
    batch_note = (
        f" (batch {batch_idx + 1} of {total_batches})" if total_batches > 1 else ""
    )

    user_content = [
        {
            "type": "text",
            "text": model_text,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": (
                f"\nAnalyze the model portfolio above for **{company_name}**{batch_note}.\n"
                "Return a JSON object matching the schema in your instructions.\n"
                "Be specific about model clusters — group models by series/purpose.\n"
                "Only return valid JSON, no markdown fencing."
            ),
        },
    ]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        temperature=0.3,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    return json.loads(raw)


SYNTHESIS_SYSTEM = """You are an AI industry analyst summarizing a company's model portfolio.

Merge multiple partial typology analyses into one concise summary.
Keep model_ids lists SHORT — at most 3 representative examples per cluster.
Focus on cluster descriptions, not exhaustive enumeration.

Return a JSON object:
{
  "company": "<name>",
  "summary": "<1-2 sentence overview>",
  "observed_types": {"general-llm": <int>, "specialized-llm": <int>, "accessory": <int>},
  "clusters": [
    {
      "label": "<cluster name>",
      "type": "general-llm|specialized-llm|accessory",
      "suggested_tags": ["tag1", "tag2"],
      "model_ids": ["up to 3 representative examples only"],
      "notes": "<optional explanation>"
    }
  ],
  "suggested_tags": ["overall list relevant to this company"]
}
Return only valid JSON. No markdown fencing."""


def synthesize_batches(
    client: anthropic.Anthropic,
    company_name: str,
    batch_results: list[dict],
) -> dict:
    """Merge multiple batch analyses into one compact company typology."""
    # Strip model_ids from batch results to keep input compact
    compact_batches = []
    for br in batch_results:
        compact = {k: v for k, v in br.items() if k != "clusters"}
        compact["clusters"] = [
            {k: v for k, v in c.items() if k != "model_ids"}
            for c in br.get("clusters", [])
        ]
        compact_batches.append(compact)

    combined_text = json.dumps(compact_batches, indent=2)
    user_content = [
        {
            "type": "text",
            "text": combined_text,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": (
                f"\nThe JSON above contains {len(batch_results)} partial analyses of "
                f"**{company_name}**'s model portfolio.\n"
                "Merge them into a single coherent company typology.\n"
                "Consolidate clusters. Keep model_ids to at most 3 examples per cluster.\n"
                "Return only valid JSON, no markdown fencing."
            ),
        },
    ]
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        temperature=0.3,
        system=SYNTHESIS_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

# Checkpoint stores two namespaces:
#   results[company_name]          = final typology (or {"error": ...})
#   _batches[company_name][b_idx]  = individual batch result (resume mid-company)

def load_checkpoint() -> tuple[dict[str, Any], dict[str, Any]]:
    if CKPT_FILE.exists():
        raw = json.loads(CKPT_FILE.read_text())
        return raw.get("results", {}), raw.get("_batches", {})
    return {}, {}


def save_checkpoint(results: dict, batches: dict) -> None:
    CKPT_FILE.write_text(json.dumps({"results": results, "_batches": batches}, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BATCH_SIZE = 100

# Graceful stop flag — set by SIGINT handler
_stop_requested = False

def _handle_sigint(sig, frame):
    global _stop_requested
    if not _stop_requested:
        print("\n[Ctrl+C] Stop requested — will save and exit after current batch.", flush=True)
        _stop_requested = True
    else:
        print("\n[Ctrl+C] Force exit.", flush=True)
        sys.exit(1)


def run(company_filter: str | None = None, limit: int | None = None) -> None:
    global _stop_requested
    signal.signal(signal.SIGINT, _handle_sigint)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    conn = get_conn()

    companies = fetch_companies(conn)
    if company_filter:
        companies = [c for c in companies if c["display_name"] == company_filter]
        if not companies:
            print(f"[ERROR] Company not found: {company_filter!r}")
            sys.exit(1)
    if limit:
        companies = companies[:limit]

    results, batch_cache = load_checkpoint()

    for i, company in enumerate(companies):
        if _stop_requested:
            print("Stopping early — progress saved.")
            break

        cname = company["display_name"]
        if cname in results and "error" not in results[cname]:
            print(f"[{i+1}/{len(companies)}] {cname} — already done, skipping")
            continue

        models = fetch_models_for_company(conn, company["id"])
        if not models:
            print(f"[{i+1}/{len(companies)}] {cname} — no models, skipping")
            results[cname] = {"company": cname, "summary": "No models.", "clusters": []}
            save_checkpoint(results, batch_cache)
            continue

        print(f"[{i+1}/{len(companies)}] {cname} — {len(models)} models", flush=True)

        try:
            if len(models) <= BATCH_SIZE:
                result = analyze_company_batch(client, cname, models)
            else:
                batches = [
                    models[j : j + BATCH_SIZE]
                    for j in range(0, len(models), BATCH_SIZE)
                ]
                # Per-batch cache keyed by company name
                company_batches: dict[str, Any] = batch_cache.get(cname, {})
                batch_results = []

                for b_idx, batch in enumerate(batches):
                    if _stop_requested:
                        # Save what we have mid-company and exit
                        batch_cache[cname] = company_batches
                        save_checkpoint(results, batch_cache)
                        print("Stopping early — batch progress saved.")
                        conn.close()
                        sys.exit(0)

                    b_key = str(b_idx)
                    if b_key in company_batches:
                        print(
                            f"  batch {b_idx+1}/{len(batches)} — already cached, skipping",
                            flush=True,
                        )
                        batch_results.append(company_batches[b_key])
                        continue

                    print(
                        f"  batch {b_idx+1}/{len(batches)} ({len(batch)} models)...",
                        flush=True,
                    )
                    br = analyze_company_batch(client, cname, batch, b_idx, len(batches))
                    batch_results.append(br)
                    company_batches[b_key] = br
                    batch_cache[cname] = company_batches
                    save_checkpoint(results, batch_cache)
                    time.sleep(1)

                if len(batch_results) == 1:
                    result = batch_results[0]
                else:
                    print("  synthesizing batches...", flush=True)
                    result = synthesize_batches(client, cname, batch_results)

                # Clear per-batch cache for this company once done
                batch_cache.pop(cname, None)

        except Exception as exc:
            print(f"  [ERROR] {exc}")
            results[cname] = {"company": cname, "error": str(exc)}
            save_checkpoint(results, batch_cache)
            continue

        results[cname] = result
        save_checkpoint(results, batch_cache)
        print(f"  done — {result.get('observed_types', {})}")
        time.sleep(0.5)

    conn.close()

    # Write final output
    output = list(results.values())
    OUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {len(output)} company typologies -> {OUT_FILE}")
    print("Next step: run 02_synthesize_typology.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: per-company typology")
    parser.add_argument("--company", help="Run for a single company (exact display_name)")
    parser.add_argument("--limit", type=int, help="Process only first N companies")
    args = parser.parse_args()
    run(company_filter=args.company, limit=args.limit)
