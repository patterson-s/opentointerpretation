#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Optional dependency only used for scraping fallback
try:
    from bs4 import BeautifulSoup  # type: ignore
    _BS4_AVAILABLE = True
except Exception:
    _BS4_AVAILABLE = False

API_BASE = "https://huggingface.co/api/models"
WEB_BASE = "https://huggingface.co"

DEFAULT_SLEEP = 0.15   # faster default, still polite
DEFAULT_RETRIES = 3
TIMEOUT = 25


# --------------------------- utilities ---------------------------

def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)
    os.replace(tmp, path)  # atomic on same filesystem

def normalize_license(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    lic = raw.strip().lower()
    lic = lic.replace("license:", "").strip()
    return lic or None

def pick_license_from_api_payload(payload: Dict[str, Any]) -> Optional[str]:
    if payload.get("license"):
        return normalize_license(payload["license"])
    card = payload.get("cardData") or {}
    for key in ("license", "License", "model_license", "model.license"):
        if key in card and card[key]:
            return normalize_license(card[key])
    tags = payload.get("tags") or []
    for t in tags:
        if isinstance(t, str) and t.lower().startswith("license:"):
            return normalize_license(t.split(":", 1)[1])
    meta = payload.get("metadata") or {}
    if meta.get("license"):
        return normalize_license(meta["license"])
    return None

def api_get_license(repo_id: str, session: requests.Session, retries: int) -> Tuple[Optional[str], Optional[str]]:
    url = f"{API_BASE}/{repo_id}"
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                return pick_license_from_api_payload(r.json()), None
            elif r.status_code == 404:
                return None, f"API 404 for {repo_id}"
            else:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(min(1.5 * attempt, 4.0))
    return None, last_err

def scrape_license(repo_id: str, session: requests.Session, retries: int) -> Tuple[Optional[str], Optional[str]]:
    if not _BS4_AVAILABLE:
        return None, "BeautifulSoup not installed"
    url = f"{WEB_BASE}/{repo_id}"
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=TIMEOUT)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code} at {url}"
            else:
                soup = BeautifulSoup(r.text, "html.parser")
                # Heuristic 1: "License:" pill then last <span>
                for tag in soup.find_all(string=lambda t: isinstance(t, str) and "License:" in t):
                    parent = getattr(tag, "parent", None)
                    if parent:
                        spans = parent.find_all("span")
                        if spans:
                            candidate = spans[-1].get_text(strip=True)
                            if candidate:
                                return normalize_license(candidate), None
                # Heuristic 2: /licenses/<slug>
                a_tags = soup.select('a[href^="/licenses/"]')
                if a_tags:
                    candidate = a_tags[0].get_text(strip=True)
                    if candidate:
                        return normalize_license(candidate), None
                return None, "License tag not found"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(min(1.5 * attempt, 4.0))
    return None, last_err

def default_out_path(in_path: Path) -> Path:
    stem = in_path.stem
    suffix = in_path.suffix or ".json"
    return in_path.with_name(f"{stem}.licensed{suffix}")

def checkpoint_paths(out_path: Path) -> Tuple[Path, Path]:
    ckpt = out_path.with_suffix(out_path.suffix + ".ckpt.json")
    miss = out_path.with_suffix(out_path.suffix + ".misses.txt")
    return ckpt, miss

def repo_id_from_record(rec: Dict[str, Any]) -> Optional[str]:
    rid = rec.get("model_id")
    if rid:
        return rid
    url = rec.get("url", "")
    if url.startswith(WEB_BASE + "/"):
        return url.replace(WEB_BASE + "/", "")
    return None

def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    return data

def index_by_repo_id(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for r in records:
        rid = repo_id_from_record(r)
        if rid:
            m[rid] = r
    return m


# --------------------------- core ---------------------------

def enrich_with_checkpoints(records: List[Dict[str, Any]],
                            out_path: Path,
                            do_scrape: bool,
                            sleep_seconds: float,
                            retries: int,
                            progress_every: int,
                            checkpoint_every: int,
                            resume: bool) -> Tuple[List[Dict[str, Any]], List[str], Counter]:
    """
    Two-pass enrichment with periodic checkpoints and resume.
    Returns (enriched_records, misses, license_counts)
    """
    # Prepare session
    session = requests.Session()
    headers = {"Accept": "application/json", "User-Agent": "license-enricher/1.3 (+https://huggingface.co)"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    session.headers.update(headers)

    # Resume support: if resume flag and an existing output/ckpt exists, merge licenses
    existing: List[Dict[str, Any]] = []
    ckpt_path, miss_path = checkpoint_paths(out_path)
    if resume:
        base_to_read = out_path if out_path.exists() else (ckpt_path if ckpt_path.exists() else None)
        if base_to_read:
            try:
                existing = load_json_list(base_to_read)
                print(f"[resume] Loaded {len(existing)} items from {base_to_read}")
            except Exception as e:
                print(f"[resume] Could not load {base_to_read}: {e}")

    # Map for quick lookup
    recs_by_id = index_by_repo_id(records)
    if existing:
        exist_by_id = index_by_repo_id(existing)
        # copy known licenses into records
        updated = 0
        for rid, rec in recs_by_id.items():
            if rid in exist_by_id:
                lic = exist_by_id[rid].get("license")
                if lic is not None and rec.get("license") != lic:
                    rec["license"] = lic
                    updated += 1
        if updated:
            print(f"[resume] Reused {updated} known licenses")

    total = len(records)
    misses: List[str] = []
    license_counts: Counter = Counter()

    # Helper to compute counts and write checkpoints periodically
    def recompute_counts() -> None:
        license_counts.clear()
        for r in records:
            license_counts[r.get("license") or "unknown"] += 1

    def write_ckpt(label: str) -> None:
        atomic_write_json(ckpt_path, records)
        # write partial miss log
        if misses:
            with open(miss_path, "w", encoding="utf-8") as f:
                f.write("\n".join(misses))
        print(f"[ckpt] {label}: wrote checkpoint -> {ckpt_path} (misses: {len(misses)})")

    # ---------------- PASS 1: API only ----------------
    print(f"[phase] API pass over {total} models …")
    processed = 0
    for idx, rec in enumerate(records, start=1):
        rid = repo_id_from_record(rec)
        if not rid:
            rec["license"] = None
            misses.append(f"[{idx}] Missing repo_id in record")
            continue

        # Skip if already resolved (resume or previous runs)
        if rec.get("license"):
            processed += 1
            if progress_every and processed % progress_every == 0:
                print(f"[API {processed}/{total}] … (skipping resolved)")
            continue

        lic, api_err = api_get_license(rid, session, retries)
        rec["license"] = lic
        processed += 1

        if progress_every and processed % progress_every == 0:
            known = sum(1 for r in records if r.get("license"))
            print(f"[API {processed}/{total}] known={known}")

        if checkpoint_every and processed % checkpoint_every == 0:
            recompute_counts()
            write_ckpt(f"API {processed}/{total}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    # ---------------- PASS 2: scrape unknowns ----------------
    unknown_idxs = [i for i, r in enumerate(records) if not r.get("license")]
    if do_scrape and _BS4_AVAILABLE and unknown_idxs:
        print(f"[phase] SCRAPE pass for {len(unknown_idxs)} unknowns …")
        for k, i in enumerate(unknown_idxs, start=1):
            rec = records[i]
            rid = repo_id_from_record(rec)
            lic, scrape_err = scrape_license(rid, session, retries)
            if lic:
                rec["license"] = lic
            else:
                misses.append(f"[{i+1}] {rid} (API miss; scrape miss: {scrape_err})")

            if progress_every and k % max(1, progress_every // 2) == 0:
                print(f"[SCRAPE {k}/{len(unknown_idxs)}] …")

            if checkpoint_every and k % checkpoint_every == 0:
                recompute_counts()
                write_ckpt(f"SCRAPE {k}/{len(unknown_idxs)}")

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    else:
        if not do_scrape:
            print("[phase] SCRAPE skipped (--no-scrape)")
        elif not _BS4_AVAILABLE:
            print("[phase] SCRAPE unavailable (install beautifulsoup4, lxml)")
        else:
            print("[phase] No unknowns to scrape 🎉")

    # final counts
    recompute_counts()
    return records, misses, license_counts


def write_final(out_path: Path, enriched: List[Dict[str, Any]], misses: List[str], license_counts: Counter) -> None:
    atomic_write_json(out_path, enriched)
    miss_log = out_path.with_suffix(out_path.suffix + ".misses.txt")
    if misses:
        with open(miss_log, "w", encoding="utf-8") as f:
            f.write("\n".join(misses))
    else:
        try:
            if miss_log.exists():
                miss_log.unlink()
        except Exception:
            pass

    print(f"\nWrote: {out_path}")
    if misses:
        print(f"{len(misses)} models missing license. Miss log: {miss_log}")
    else:
        print("All models have a license value.")

    # breakdown
    print("\nLicense breakdown (top first):")
    total = sum(license_counts.values())
    for lic, cnt in license_counts.most_common():
        pct = (cnt / total) * 100 if total else 0
        print(f"  {lic:>12}: {cnt:>5}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':>12}: {total:>5}")


# --------------------------- CLI ---------------------------

def main():
    p = argparse.ArgumentParser(description="Add `license` to HF model-card JSON (checkpoints + resume).")
    p.add_argument("input_json", type=str, help="Path to input JSON (list of model records).")
    p.add_argument("--out", type=str, default=None, help="Output JSON (default: *.licensed.json).")
    p.add_argument("--no-scrape", action="store_true", help="Disable HTML fallback scraping.")
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="Seconds to sleep between requests.")
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="HTTP retry attempts.")
    p.add_argument("--progress", type=int, default=200, help="Print a progress tick every N records (0 = silent).")
    p.add_argument("--checkpoint-every", type=int, default=200, help="Write checkpoint every N items (0 = off).")
    p.add_argument("--resume", action="store_true", help="Resume from existing output/ckpt if present.")
    args = p.parse_args()

    in_path = Path(args.input_json)
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else default_out_path(in_path)

    try:
        records = load_json_list(in_path)
    except Exception as e:
        print(f"Failed to load JSON: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} records from: {in_path}")
    print(f"Writing to: {out_path}")
    if args.no_scrape:
        print("HTML fallback: DISABLED")
    else:
        if not _BS4_AVAILABLE:
            print("HTML fallback: unavailable (pip install beautifulsoup4 lxml)")
        else:
            print("HTML fallback: ENABLED")

    try:
        enriched, misses, license_counts = enrich_with_checkpoints(
            records=records,
            out_path=out_path,
            do_scrape=(not args.no_scrape),
            sleep_seconds=max(0.0, args.sleep),
            retries=max(1, args.retries),
            progress_every=max(0, args.progress),
            checkpoint_every=max(0, args.checkpoint_every),
            resume=args.resume,
        )
    except KeyboardInterrupt:
        # last-ditch checkpoint
        ckpt_path, miss_path = checkpoint_paths(out_path)
        print("\nInterrupted — writing final checkpoint…")
        atomic_write_json(ckpt_path, records)
        if 'misses' in locals() and misses:
            with open(miss_path, "w", encoding="utf-8") as f:
                f.write("\n".join(misses))
        print(f"Checkpoint written to: {ckpt_path}")
        sys.exit(130)

    write_final(out_path, enriched, misses, license_counts)


if __name__ == "__main__":
    main()
