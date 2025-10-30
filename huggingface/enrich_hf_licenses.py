#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import html
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

HF_API = "https://huggingface.co/api/models/"
HTTP_TIMEOUT = 20
UA = {"User-Agent": "opentointerpretation-license-bot/2.0"}

LICENSE_SLUG_MAP = {
    # Common normalizations
    "apache 2.0": "apache-2.0",
    "apache-2": "apache-2.0",
    "apache-2.0": "apache-2.0",
    "apache license 2.0": "apache-2.0",
    "apache license version 2.0": "apache-2.0",
    "mit": "mit",
    "bsd-3-clause": "bsd-3-clause",
    "bsd-2-clause": "bsd-2-clause",
    "mpl-2.0": "mpl-2.0",
    "agpl-3.0": "agpl-3.0",
    "gpl-3.0": "gpl-3.0",
    "lgpl-3.0": "lgpl-3.0",
    "cc-by-4.0": "cc-by-4.0",
    "cc-by-sa-4.0": "cc-by-sa-4.0",
    "cc-by-nc-4.0": "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0": "cc-by-nc-sa-4.0",
    "openrail": "openrail",
    "openrail++": "openrail++",
    "bigscience-bloom-rail-1.0": "bigscience-bloom-rail-1.0",
    "llama 3 community license": "llama3-community",
    "llama 2 community license": "llama2-community",
    # Some pages show "other" or "unknown" – keep as-is if found.
}

LICENSE_FINGERPRINTS = [
    (re.compile(r"apache\s+license\s+version\s*2\.0", re.I), "apache-2.0"),
    (re.compile(r"\bmit\s+license\b", re.I), "mit"),
    (re.compile(r"\bmozilla public license\s*version\s*2\.0\b", re.I), "mpl-2.0"),
    (re.compile(r"\bgnu\s+general\s+public\s+license\s+version\s*3\b", re.I), "gpl-3.0"),
    (re.compile(r"\bgnu\s+lesser\s+general\s+public\s+license\s+version\s*3\b", re.I), "lgpl-3.0"),
    (re.compile(r"\baffero\s+general\s+public\s+license\s+version\s*3\b", re.I), "agpl-3.0"),
    (re.compile(r"\bbsd\s*(?:3|three)[-\s]*clause\b", re.I), "bsd-3-clause"),
    (re.compile(r"\bcc[-\s]*by[-\s]*nc[-\s]*sa[-\s]*4\.0\b", re.I), "cc-by-nc-sa-4.0"),
    (re.compile(r"\bcc[-\s]*by[-\s]*nc[-\s]*4\.0\b", re.I), "cc-by-nc-4.0"),
    (re.compile(r"\bcc[-\s]*by[-\s]*sa[-\s]*4\.0\b", re.I), "cc-by-sa-4.0"),
    (re.compile(r"\bcc[-\s]*by[-\s]*4\.0\b", re.I), "cc-by-4.0"),
]

# ---------- Utilities ----------

def normalize_license(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    slug = re.sub(r"\s+", " ", s.strip()).lower()
    slug = slug.replace("_", "-")
    slug = LICENSE_SLUG_MAP.get(slug, slug)
    # Coerce some obvious variants:
    slug = re.sub(r"apache\s*(?:licen[cs]e)?\s*version?\s*2(?:\.0)?", "apache-2.0", slug)
    slug = re.sub(r"\bmit(?=\b)", "mit", slug)
    slug = re.sub(r"\bmpl\s*2(?:\.0)?\b", "mpl-2.0", slug)
    slug = re.sub(r"\bcc\s*by\s*nc\s*sa\s*4(?:\.0)?\b", "cc-by-nc-sa-4.0", slug)
    slug = re.sub(r"\bcc\s*by\s*nc\s*4(?:\.0)?\b", "cc-by-nc-4.0", slug)
    slug = re.sub(r"\bcc\s*by\s*sa\s*4(?:\.0)?\b", "cc-by-sa-4.0", slug)
    slug = re.sub(r"\bcc\s*by\s*4(?:\.0)?\b", "cc-by-4.0", slug)
    return slug


def checkpoint_paths(out_path: Path) -> Tuple[Path, Path]:
    return Path(str(out_path) + ".ckpt.json"), Path(str(out_path) + ".misses.txt")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get(repo_id: str, url: str, **kw) -> requests.Response:
    return requests.get(url, timeout=HTTP_TIMEOUT, headers=UA, **kw)


# ---------- Extractors ----------

def license_from_api(repo_id: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        r = get(repo_id, HF_API + repo_id)
        if r.status_code != 200:
            return None, f"API {r.status_code}"
        data = r.json()
        # 1) direct license
        cand = data.get("license")
        if cand:
            return normalize_license(cand), None
        # 2) cardData.license
        card = data.get("cardData") or {}
        cand = card.get("license")
        if cand:
            return normalize_license(cand), None
        # 3) metadata.license
        md = data.get("metadata") or {}
        cand = md.get("license")
        if cand:
            return normalize_license(cand), None
        # 4) tags with "license:<slug>"
        tags = data.get("tags") or []
        for t in tags:
            m = re.match(r"license\s*:\s*(.+)", str(t), re.I)
            if m:
                return normalize_license(m.group(1)), None
        # 5) look for LICENSE sibling
        siblings = data.get("siblings") or []
        for sib in siblings:
            name = sib.get("rfilename") or ""
            if name.upper() in {"LICENSE", "LICENSE.TXT"}:
                # fingerprint below (fallback 5 handles download)
                pass
        return None, "API miss"
    except Exception as e:
        return None, f"API error: {e}"


def license_from_static_html(repo_url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse server-rendered HTML for 'License:' chip or /licenses/<slug> link.
    (Works when HF still emits those in SSR; may miss for JS-only pages.)
    """
    try:
        r = get(repo_url, repo_url)
        if r.status_code != 200:
            return None, f"scrape {r.status_code}"
        html_text = r.text
        # Heuristic 1: "License:" followed by a badge-like span/a nearby
        m = re.search(r">License:\s*</[^>]+>\s*([^<]+)<", html_text, re.I)
        if m:
            return normalize_license(html.unescape(m.group(1)).strip()), None
        # Heuristic 2: /licenses/<slug> anchor
        m = re.search(r'href="/licenses/([^"?/#\s]+)"', html_text, re.I)
        if m:
            return normalize_license(m.group(1)), None
        return None, "scrape miss: License tag not found"
    except Exception as e:
        return None, f"scrape error: {e}"


def license_from_hydration_json(repo_url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse preloaded JSON from a <script> tag used to hydrate the React app.
    We search for a JSON substring that contains `"cardData"` and `"license"`.
    """
    try:
        r = get(repo_url, repo_url)
        if r.status_code != 200:
            return None, f"hydrate {r.status_code}"
        html_text = r.text
        # Extract the biggest JSON-like blob that contains "cardData"
        # Then find "license":"..."
        # Keep it regex-based to avoid strict JSON parsing on huge blobs.
        block = None
        for m in re.finditer(r"<script[^>]*>(.*?)</script>", html_text, re.S | re.I):
            s = m.group(1)
            if "cardData" in s and "license" in s:
                block = s
                break
        if not block:
            return None, "hydrate miss: script with cardData not found"
        m = re.search(r'"license"\s*:\s*"([^"]+)"', block)
        if not m:
            return None, "hydrate miss: license not in script"
        return normalize_license(html.unescape(m.group(1))), None
    except Exception as e:
        return None, f"hydrate error: {e}"


def _readme_front_matter(text: str) -> Dict[str, str]:
    """
    Parse very simple YAML front-matter: key: value lines between first two ---.
    Avoid PyYAML dependency; we only need 'license:'.
    """
    m = re.match(r"\s*---\s*\r?\n(.*?)\r?\n---\s*\r?\n", text, re.S)
    if not m:
        return {}
    fm = m.group(1)
    result: Dict[str, str] = {}
    for line in fm.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            result[k.strip().lower()] = v.strip().strip('"').strip("'")
    return result


def license_from_raw_readme(repo_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch https://huggingface.co/<repo>/raw/README.md and parse YAML front-matter.
    """
    url = f"https://huggingface.co/{repo_id}/raw/README.md"
    try:
        r = get(repo_id, url)
        if r.status_code != 200:
            return None, f"readme {r.status_code}"
        fm = _readme_front_matter(r.text or "")
        cand = fm.get("license")
        if cand:
            return normalize_license(cand), None
        return None, "readme miss: no license in front matter"
    except Exception as e:
        return None, f"readme error: {e}"


def license_from_license_file(repo_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Use the API to list siblings, fetch LICENSE if present, then fingerprint text.
    """
    try:
        r = get(repo_id, HF_API + repo_id)
        if r.status_code != 200:
            return None, f"licensefile api {r.status_code}"
        siblings = (r.json() or {}).get("siblings") or []
        lic_name = None
        for sib in siblings:
            name = (sib.get("rfilename") or "").strip()
            if name.upper() in {"LICENSE", "LICENSE.TXT"}:
                lic_name = name
                break
        if not lic_name:
            return None, "licensefile miss: no LICENSE blob"
        raw = f"https://huggingface.co/{repo_id}/resolve/main/{lic_name}"
        rr = get(repo_id, raw, allow_redirects=True)
        if rr.status_code != 200:
            return None, f"licensefile fetch {rr.status_code}"
        txt = rr.text or ""
        for pat, slug in LICENSE_FINGERPRINTS:
            if pat.search(txt):
                return slug, None
        return None, "licensefile miss: unknown license text"
    except Exception as e:
        return None, f"licensefile error: {e}"


# ---------- Processing ----------

@dataclass
class Outcome:
    license: Optional[str]
    note: Optional[str]


def resolve_license(repo_id: str, repo_url: str) -> Outcome:
    # 1) API
    lic, note = license_from_api(repo_id)
    if lic:
        return Outcome(lic, "api")
    # 2) Static HTML
    lic, note2 = license_from_static_html(repo_url)
    if lic:
        return Outcome(lic, "scrape")
    # 3) Hydration JSON
    lic, note3 = license_from_hydration_json(repo_url)
    if lic:
        return Outcome(lic, "hydrate")
    # 4) Raw README.md front matter
    lic, note4 = license_from_raw_readme(repo_id)
    if lic:
        return Outcome(lic, "readme")
    # 5) LICENSE file fingerprint
    lic, note5 = license_from_license_file(repo_id)
    if lic:
        return Outcome(lic, "licensefile")
    # Construct combined note
    reasons = "; ".join(
        r for r in [note, note2, note3, note4, note5] if r
    ) or "unknown"
    return Outcome(None, reasons)


def enrich(records: List[Dict], interval: int = 100, verbose: bool = True
           ) -> Tuple[List[Dict], List[str]]:
    """
    For each record, if 'license' is missing/empty, attempt to resolve it.
    Returns (enriched_records, misses_list)
    """
    out: List[Dict] = []
    misses: List[str] = []
    t0 = time.time()

    for i, rec in enumerate(records):
        repo_id = rec.get("model_id") or rec.get("repo_id")
        url = rec.get("url") or (f"https://huggingface.co/{repo_id}" if repo_id else None)

        if not repo_id or not url:
            out.append(rec)
            misses.append(f"[{i}] <missing repo_id/url> (skip)")
            continue

        current = (rec.get("license") or "").strip()
        if current:
            out.append(rec)
            if verbose and i % interval == 0:
                print(f"[{i}] {repo_id}: already has license ({current})")
            continue

        oc = resolve_license(repo_id, url)
        if oc.license:
            newrec = dict(rec)
            newrec["license"] = oc.license
            newrec["_license_source"] = oc.note  # provenance for debugging
            out.append(newrec)
            if verbose:
                print(f"[{i}] {repo_id}: {oc.license} ({oc.note})")
        else:
            out.append(rec)
            entry = f"[{i}] {repo_id} ({oc.note})"
            misses.append(entry)
            if verbose:
                print(f"[{i}] {repo_id}: MISS -> {oc.note}")

        # checkpoint
        if (i + 1) % interval == 0:
            yield out, misses, i + 1, time.time() - t0

    yield out, misses, len(records), time.time() - t0


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Enrich HF licenses with robust fallbacks.")
    ap.add_argument("input", help="Path to input JSON array")
    ap.add_argument("output", help="Path to write enriched JSON")
    ap.add_argument("--interval", type=int, default=100, help="Checkpoint interval (default 100)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    ckpt_path, miss_path = checkpoint_paths(out_path)

    records = load_json(in_path)
    print(f"Loaded {len(records)} records from: {in_path}")
    print(f"Writing to: {out_path}")
    print("HTML fallback: ENABLED (static + hydration + README + LICENSE)")

    try:
        last_out, last_miss = None, None
        for out, misses, n, elapsed in enrich(records, interval=args.interval):
            write_json(ckpt_path, out)
            with open(miss_path, "w", encoding="utf-8") as f:
                f.write("\n".join(misses))
            print(f"Checkpoint @ {n} ({elapsed:.1f}s): {ckpt_path.name} / {miss_path.name}")
            last_out, last_miss = out, misses

    except KeyboardInterrupt:
        print("\nInterrupted — leaving latest checkpoint on disk.")
        return

    # finalize
    write_json(out_path, last_out)
    with open(miss_path, "w", encoding="utf-8") as f:
        f.write("\n".join(last_miss))
    print(f"\nDone.\nFinal: {out_path}\nMisses: {miss_path}")


if __name__ == "__main__":
    main()
