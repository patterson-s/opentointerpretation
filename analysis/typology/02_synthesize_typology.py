#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2: Cross-company typology synthesis using Claude Sonnet.

Reads company_typologies.json (output of Phase 1) and derives a single
canonical typology scheme covering all companies — finalized types,
tag vocabulary, and definitions with examples.

Output: analysis/typology/typology_scheme.json  (human-editable before Phase 3)

Usage:
    python analysis/typology/02_synthesize_typology.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "analysis" / "typology"
IN_FILE = OUT_DIR / "company_typologies.json"
OUT_FILE = OUT_DIR / "typology_scheme.json"

load_dotenv(ROOT / ".env", override=True)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI industry analyst building a canonical taxonomy
of AI model types across frontier AI companies.

The taxonomy must be:
1. **Covering** — every model can be classified
2. **Discriminating** — types are meaningfully distinct
3. **Stable** — types should not change as new models are added

Type framework (mutually exclusive):
- general-llm: broad-capability language/chat/completion models; no strong domain focus
- specialized-llm: LMs with a primary specialization (code, math, vision, audio, reasoning, etc.)
- accessory: non-generation models that support ML pipelines (embeddings, rerankers,
  classifiers, ASR, image encoders, segmentation, depth, OCR, etc.)

Tag vocabulary (multiple, overlapping — descriptive, not exclusive):
Tags should capture the key axes of variation: capability focus, training stage,
architecture variant, use-case context. Minimize redundancy across tags.

Return a JSON object with this exact schema:
{
  "version": "1.0",
  "types": [
    {
      "id": "general-llm",
      "label": "General LLM",
      "definition": "<1-2 sentences>",
      "examples": ["model_id1", "model_id2"]
    },
    ...
  ],
  "tags": [
    {
      "id": "tag-id",
      "label": "Tag Label",
      "definition": "<1 sentence>",
      "example_models": ["model_id1"]
    },
    ...
  ],
  "notes": "<Any important guidance for applying this typology consistently>"
}"""


def run() -> None:
    if not IN_FILE.exists():
        print(f"[ERROR] Input file not found: {IN_FILE}")
        print("Run 01_per_company_typology.py first.")
        sys.exit(1)

    company_typologies = json.loads(IN_FILE.read_text())
    print(f"Loaded {len(company_typologies)} company typologies from {IN_FILE}")

    # Build a compact summary of each company for the prompt
    summaries = []
    all_tags: list[str] = []
    for ct in company_typologies:
        if "error" in ct:
            continue
        # Strip model_ids from clusters to keep input compact
        compact = {k: v for k, v in ct.items() if k != "clusters"}
        compact["clusters"] = [
            {k: v for k, v in c.items() if k != "model_ids"}
            for c in ct.get("clusters", [])
        ]
        summaries.append(compact)
        all_tags.extend(ct.get("suggested_tags", []))

    # Deduplicate tags for reference
    unique_tags = sorted(set(t.lower() for t in all_tags))
    tag_hint = ", ".join(unique_tags[:80])

    typology_text = json.dumps(summaries, indent=2)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print("Sending to Claude Sonnet for synthesis...", flush=True)

    user_content = [
        {
            "type": "text",
            "text": typology_text,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": (
                "\nThe JSON above contains per-company typology analyses for "
                f"{len(summaries)} frontier AI companies.\n\n"
                "Your task: synthesize a single **canonical typology scheme** that:\n"
                "1. Covers all model types observed across all companies\n"
                "2. Uses the three-type framework: general-llm / specialized-llm / accessory\n"
                "3. Defines a compact, non-redundant tag vocabulary\n\n"
                f"Tags seen across companies (for reference, not exhaustive): {tag_hint}\n\n"
                "Return only the JSON schema described in your instructions. No markdown fencing."
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

    scheme = json.loads(raw)
    OUT_FILE.write_text(json.dumps(scheme, indent=2))

    # Print summary for user review
    print(f"\n{'='*60}")
    print("TYPOLOGY SCHEME SUMMARY")
    print(f"{'='*60}")
    print(f"\nTypes ({len(scheme.get('types', []))}):")
    for t in scheme.get("types", []):
        print(f"  [{t['id']}] {t['label']}")
        print(f"    {t['definition']}")

    print(f"\nTags ({len(scheme.get('tags', []))}):")
    for tag in scheme.get("tags", []):
        defn = tag['definition'].encode('ascii', 'replace').decode('ascii')
        print(f"  [{tag['id']}] {tag['label']} -- {defn}")

    if scheme.get("notes"):
        print(f"\nNotes: {scheme['notes']}")

    print(f"\n{'='*60}")
    print(f"Written to: {OUT_FILE}")
    print(
        "\nReview and edit typology_scheme.json if needed, then run:\n"
        "  python analysis/typology/03_classify_models.py"
    )


if __name__ == "__main__":
    run()
