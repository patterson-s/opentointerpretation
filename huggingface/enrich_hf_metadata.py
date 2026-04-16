#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enrich HuggingFace model records with additional metadata from the HF REST API.

For each model, fetches https://huggingface.co/api/models/{model_id} and extracts:
  - pipeline_tag      → metadata.pipeline_tag
  - derived modality  → metadata.modality
  - num_parameters    → metadata.num_parameters (billions, float, from safetensors)
  - createdAt         → release_date (ISO 8601 string)
  - lastModified      → metadata.last_modified
  - library_name      → metadata.library_name
  - gated             → metadata.gated
  - cardData.language → metadata.language
  - config.architectures → metadata.architectures
  - config.model_type    → metadata.model_type
  - tags              → metadata.tags (filtered)
  - cardData.base_model       → metadata.base_model
  - cardData.fine-tuned-from  → metadata.fine_tuned_from
  - cardData.datasets         → metadata.datasets
  - cardData.metrics          → metadata.eval_metrics
  - trendingScore             → metadata.trending_score
  - inference                 → metadata.inference
  - config.vocab_size         → metadata.vocab_size
  - config.hidden_size        → metadata.hidden_size
  - config.num_hidden_layers  → metadata.num_hidden_layers
  - config.num_attention_heads→ metadata.num_attention_heads
  - config.max_position_embeddings → metadata.max_position_embeddings
  - safetensors.total         → metadata.safetensors_total
  - siblings (derived)        → metadata.has_license_file, metadata.has_readme
  - disabled                  → metadata.disabled
  - private                   → metadata.private

Usage:
    python huggingface/enrich_hf_metadata.py \\
        --input huggingface/hf_6jan2026.json \\
        --output huggingface/hf_6jan2026_meta.json \\
        [--interval 100] [--delay 0.3] [--limit 10]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

HF_API = "https://huggingface.co/api/models/"
HTTP_TIMEOUT = 20
UA = {"User-Agent": "opentointerpretation-metadata-bot/1.0"}

# Map pipeline_tag → modality category
PIPELINE_TO_MODALITY: Dict[str, str] = {
    # text
    "text-generation": "text",
    "text2text-generation": "text",
    "fill-mask": "text",
    "token-classification": "text",
    "text-classification": "text",
    "question-answering": "text",
    "summarization": "text",
    "translation": "text",
    "sentence-similarity": "text",
    "feature-extraction": "text",
    "zero-shot-classification": "text",
    # multimodal (vision + language)
    "image-to-text": "multimodal",
    "visual-question-answering": "multimodal",
    "image-text-to-text": "multimodal",
    "document-question-answering": "multimodal",
    "video-text-to-text": "multimodal",
    # vision
    "text-to-image": "vision",
    "image-classification": "vision",
    "image-segmentation": "vision",
    "object-detection": "vision",
    "depth-estimation": "vision",
    "image-to-image": "vision",
    "image-feature-extraction": "vision",
    # audio
    "automatic-speech-recognition": "audio",
    "text-to-audio": "audio",
    "text-to-speech": "audio",
    "audio-classification": "audio",
    "audio-to-audio": "audio",
    # video
    "text-to-video": "video",
    "image-to-video": "video",
}


# ---------- HTTP ----------

def fetch_hf_model_api(model_id: str) -> Optional[Dict]:
    """GET /api/models/{model_id}. Returns parsed JSON or None on error."""
    try:
        r = requests.get(
            HF_API + model_id,
            timeout=HTTP_TIMEOUT,
            headers=UA,
        )
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


# ---------- Extraction ----------

def derive_modality(pipeline_tag: Optional[str]) -> Optional[str]:
    if not pipeline_tag:
        return None
    return PIPELINE_TO_MODALITY.get(pipeline_tag.strip().lower())


def derive_num_parameters(safetensors: Optional[Dict]) -> Optional[float]:
    """
    safetensors.parameters is a dict like {"BF16": 7241732096} where values
    are parameter counts (not bytes). Sum all dtype counts and convert to billions.
    Returns None if no safetensors data is available.
    """
    if not safetensors:
        return None
    params = safetensors.get("parameters")
    if not params or not isinstance(params, dict):
        return None
    total = sum(v for v in params.values() if isinstance(v, (int, float)))
    if total <= 0:
        return None
    return round(total / 1e9, 4)


def extract_metadata_fields(api: Dict) -> Dict[str, Any]:
    """Parse the HF API response and return a flat dict of new fields."""
    pipeline_tag = api.get("pipeline_tag") or None
    card_data = api.get("cardData") or {}
    config = api.get("config") or {}
    safetensors = api.get("safetensors") or {}

    # Tags: strip noise, keep language codes, arxiv refs, and task tags
    raw_tags: List[str] = api.get("tags") or []
    filtered_tags = [
        t for t in raw_tags
        if not t.startswith("license:") and t not in ("transformers", "pytorch", "safetensors")
    ]

    # Language: prefer cardData.language list
    language = card_data.get("language") or None
    if isinstance(language, str):
        language = [language]

    # Architectures list
    architectures = config.get("architectures") or None

    # Siblings: scan once to derive file presence flags
    siblings: List[Dict] = api.get("siblings") or []
    sibling_names = {(s.get("rfilename") or "").upper() for s in siblings}
    has_license_file = bool(sibling_names & {"LICENSE", "LICENSE.TXT", "LICENSE.MD"})
    has_readme = "README.MD" in sibling_names

    fields: Dict[str, Any] = {
        # Existing fields
        "pipeline_tag": pipeline_tag,
        "modality": derive_modality(pipeline_tag),
        "num_parameters": derive_num_parameters(safetensors if safetensors else None),
        "release_date": api.get("createdAt") or None,
        "last_modified": api.get("lastModified") or None,
        "library_name": api.get("library_name") or None,
        "gated": api.get("gated"),        # False | "manual" | "auto"
        "language": language,
        "architectures": architectures,
        "model_type": config.get("model_type") or None,
        "tags": filtered_tags or None,
        # New: model lineage
        "base_model": card_data.get("base_model") or None,
        "fine_tuned_from": card_data.get("fine-tuned-from") or None,
        # New: training/eval metadata from card front-matter
        "datasets": card_data.get("datasets") or None,
        "eval_metrics": card_data.get("metrics") or None,
        # New: HF platform signals
        "trending_score": api.get("trendingScore") or None,
        "inference": api.get("inference") if api.get("inference") is not None else None,
        # New: architecture hyperparameters
        "vocab_size": config.get("vocab_size") or None,
        "hidden_size": config.get("hidden_size") or None,
        "num_hidden_layers": config.get("num_hidden_layers") or None,
        "num_attention_heads": config.get("num_attention_heads") or None,
        "max_position_embeddings": config.get("max_position_embeddings") or None,
        # New: file/size info
        "safetensors_total": safetensors.get("total") or None,
        "has_license_file": has_license_file,
        "has_readme": has_readme,
        # New: model status
        "disabled": api.get("disabled") if api.get("disabled") is not None else None,
        "private": api.get("private") if api.get("private") is not None else None,
    }
    # Remove keys that are None so we don't overwrite existing data with nulls.
    # Note: boolean False values (gated=False, disabled=False, etc.) are kept
    # because False is not None.
    return {k: v for k, v in fields.items() if v is not None}


# ---------- Checkpoint helpers ----------

def checkpoint_paths(out_path: Path) -> Tuple[Path, Path]:
    return Path(str(out_path) + ".ckpt.json"), Path(str(out_path) + ".misses.txt")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Enrichment loop ----------

def enrich(
    records: List[Dict],
    interval: int = 100,
    delay: float = 0.3,
    verbose: bool = True,
):
    """
    Generator that enriches records by fetching HF API for each model.
    Yields (out, misses, n_processed, elapsed_seconds) every `interval` records
    and once at the end.
    """
    out: List[Dict] = []
    misses: List[str] = []
    t0 = time.time()

    for i, rec in enumerate(records):
        model_id = rec.get("model_id") or rec.get("repo_id")
        if not model_id:
            out.append(rec)
            misses.append(f"[{i}] <missing model_id> — skipped")
            continue

        api_data = fetch_hf_model_api(model_id)
        updated = dict(rec)

        if api_data is None:
            misses.append(f"[{i}] {model_id}: API fetch failed")
            if verbose:
                print(f"[{i}] {model_id}: MISS (API fetch failed)")
        else:
            new_fields = extract_metadata_fields(api_data)
            updated.update(new_fields)
            if verbose and (i + 1) % interval == 0:
                print(
                    f"[{i+1}] {model_id}: "
                    f"tag={new_fields.get('pipeline_tag')} "
                    f"modality={new_fields.get('modality')} "
                    f"params={new_fields.get('num_parameters')}B"
                )

        out.append(updated)

        if delay > 0:
            time.sleep(delay)

        if (i + 1) % interval == 0:
            yield out, misses, i + 1, time.time() - t0

    yield out, misses, len(records), time.time() - t0


# ---------- CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Enrich HF model records with pipeline_tag, modality, parameters, release_date, etc."
    )
    ap.add_argument("--input",  required=True, help="Path to input JSON array")
    ap.add_argument("--output", required=True, help="Path to write enriched JSON")
    ap.add_argument("--interval", type=int,   default=100,  help="Checkpoint interval (default 100)")
    ap.add_argument("--delay",    type=float, default=0.3,  help="Seconds between API calls (default 0.3)")
    ap.add_argument("--limit",    type=int,   default=None, help="Process only first N records (for testing)")
    args = ap.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)
    ckpt_path, miss_path = checkpoint_paths(out_path)

    records = load_json(in_path)
    if args.limit:
        records = records[: args.limit]

    print(f"Loaded {len(records)} records from: {in_path}")
    print(f"Writing to: {out_path}")
    print(f"Fields: pipeline_tag, modality, num_parameters, release_date, last_modified, "
          f"library_name, gated, language, architectures, model_type, tags, "
          f"base_model, fine_tuned_from, datasets, eval_metrics, trending_score, inference, "
          f"vocab_size, hidden_size, num_hidden_layers, num_attention_heads, "
          f"max_position_embeddings, safetensors_total, has_license_file, has_readme, "
          f"disabled, private")
    print(f"Delay: {args.delay}s between calls | Checkpoint every {args.interval} records")
    if args.limit:
        print(f"Limit: processing first {args.limit} records only")
    print()

    try:
        last_out = last_miss = None
        for out, misses, n, elapsed in enrich(
            records, interval=args.interval, delay=args.delay
        ):
            write_json(ckpt_path, out)
            with open(miss_path, "w", encoding="utf-8") as f:
                f.write("\n".join(misses))
            rate = n / elapsed if elapsed > 0 else 0
            print(f"Checkpoint @ {n}/{len(records)} ({elapsed:.1f}s, {rate:.1f} rec/s)")
            last_out, last_miss = out, misses

    except KeyboardInterrupt:
        print("\nInterrupted — checkpoint preserved on disk.")
        return

    write_json(out_path, last_out)
    with open(miss_path, "w", encoding="utf-8") as f:
        f.write("\n".join(last_miss or []))

    miss_count = len(last_miss) if last_miss else 0
    print(f"\nDone. {len(last_out)} records written to: {out_path}")
    print(f"Misses: {miss_count} (see {miss_path.name})")


if __name__ == "__main__":
    main()
