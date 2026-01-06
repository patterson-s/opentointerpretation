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
UA = {"User-Agent": "opentointerpretation-license-bot/2.1"}

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

# --- Vendor mapping for "other" ---
VENDOR_URL_TO_SLUG = [
    (re.compile(r"(?:^|//)(?:www\.)?nvidia\.com|developer\.nvidia\.com", re.I), "proprietary:nvidia-aiml"),
    (re.compile(r"(?:^|//)(?:www\.)?stability\.ai|stability\.ai/.*/license", re.I), "proprietary:stability-terms"),
    (re.compile(r"(?:^|//)(?:www\.)?microsoft\.com|aka\.ms/", re.I), "proprietary:microsoft-research"),
    (re.compile(r"(?:^|//)(?:www\.)?openai\.com", re.I), "proprietary:openai-terms"),
    (re.compile(r"(?:^|//)(?:www\.)?anthropic\.com", re.I), "proprietary:anthropic-terms"),
    (re.compile(r"(?:^|//)(?:www\.)?x\.com|(?:^|//)(?:www\.)?xai\.com", re.I), "proprietary:xai-terms"),
    (re.compile(r"(?:^|//)(?:www\.)?google\.com|ai\.google|deepmind\.com", re.I), "proprietary:google-ai-terms"),
]

OTHER_KEYWORDS = re.compile(
    r"\b(license|model\s*license|terms\s*of\s*use|terms|policy|eula|acceptable\s*use)\b",
    re.I,
)

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


# ---------- Extractors (existing) ----------

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
        # (download/fingerprint handled in fallback)
        return None, "API miss"
    except Exception as e:
        return None, f"API error: {e}"


def license_from_static_html(repo_url: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        r = get(repo_url, repo_url)
        if r.status_code != 200:
            return None, f"scrape {r.status_code}"
        html_text = r.text
        m = re.search(r">License:\s*</[^>]+>\s*([^<]+)<", html_text, re.I)
        if m:
            return normalize_license(html.unescape(m.group(1)).strip()), None
        m = re.search(r'href="/licenses/([^"?/#\s]+)"', html_text, re.I)
        if m:
            return normalize_license(m.group(1)), None
        return None, "scrape miss: License tag not found"
    except Exception as e:
        return None, f"scrape error: {e}"


def license_from_hydration_json(repo_url: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        r = get(repo_url, repo_url)
        if r.status_code != 200:
            return None, f"hydrate {r.status_code}"
        html_text = r.text
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


# ---------- “other” resolution helpers ----------

def _extract_markdown_links(text: str) -> List[Tuple[str, str]]:
    # [(link_text, link_url)]
    links = []
    for m in re.finditer(r"\[([^\]]+)\]\(([^\s)]+)\)", text):
        links.append((m.group(1), m.group(2)))
    return links


def _score_link(text: str, url: str) -> int:
    score = 0
    if OTHER_KEYWORDS.search(text or ""):
        score += 40
    if OTHER_KEYWORDS.search(url or ""):
        score += 40
    # prefer vendor domains we know
    for rx, _slug in VENDOR_URL_TO_SLUG:
        if rx.search(url or ""):
            score += 20
    return score


def _map_vendor_slug(url: str) -> Optional[str]:
    for rx, slug in VENDOR_URL_TO_SLUG:
        if rx.search(url or ""):
            return slug
    return None


def readme_body_candidates(repo_id: str) -> List[Tuple[str, str, int]]:
    """Return [(name, url, score)] from README body (not just front-matter)."""
    url = f"https://huggingface.co/{repo_id}/raw/README.md"
    try:
        r = get(repo_id, url)
        if r.status_code != 200:
            return []
        text = r.text or ""
        links = _extract_markdown_links(text)
        cands = []
        for name, href in links:
            s = _score_link(name, href)
            if s > 0:
                cands.append((name.strip(), href.strip(), s))
        # also scan plain URLs near 'license/terms' keywords
        for m in re.finditer(r"(https?://[^\s)]+)", text):
            href = m.group(1)
            # look back/forward 120 chars for keywords
            start = max(0, m.start() - 120)
            end = min(len(text), m.end() + 120)
            window = text[start:end]
            s = _score_link(window, href)
            if s > 0:
                cands.append(("(inline)", href.strip(), s))
        return sorted(cands, key=lambda x: x[2], reverse=True)
    except Exception:
        return []


def hydration_url_candidates(repo_url: str) -> List[Tuple[str, int]]:
    """Return [(url, score)] from hydration scripts on the HTML page."""
    try:
        r = get(repo_url, repo_url)
        if r.status_code != 200:
            return []
        html_text = r.text or ""
        urls = set(re.findall(r"https?://[^\s\"'<>)]+", html_text))
        cands = []
        for href in urls:
            s = _score_link(href, href)
            if s > 0:
                cands.append((href.strip(), s))
        return sorted(cands, key=lambda x: x[1], reverse=True)
    except Exception:
        return []


def sibling_terms_candidates(repo_id: str) -> List[Tuple[str, str, int]]:
    """
    Look for TERMS / EULA / MODEL_LICENSE type files at repo root and return synthetic
    links using the raw URL.
    """
    try:
        r = get(repo_id, HF_API + repo_id)
        if r.status_code != 200:
            return []
        siblings = (r.json() or {}).get("siblings") or []
        targets = []
        for sib in siblings:
            name = (sib.get("rfilename") or "").strip()
            upper = name.upper()
            if upper in {"TERMS.md", "TERMS.txt", "TERMS", "EULA.md", "EULA.txt", "MODEL_LICENSE", "MODEL_LICENSE.md"}:
                raw = f"https://huggingface.co/{repo_id}/resolve/main/{name}"
                targets.append((name, raw))
        cands = []
        for name, href in targets:
            s = _score_link(name, href) + 15  # small boost for explicit file
            cands.append((name, href, s))
        return sorted(cands, key=lambda x: x[2], reverse=True)
    except Exception:
        return []


@dataclass
class OtherOutcome:
    license_slug: Optional[str]
    license_name: Optional[str]
    license_url: Optional[str]
    license_family: Optional[str]
    license_confidence: int
    note: str


def resolve_other_license(repo_id: str, repo_url: str) -> OtherOutcome:
    """
    Try to turn 'other' into a named proprietary license by harvesting a (name, url)
    and mapping to a vendor slug. Non-fatal; returns best-effort metadata.
    """
    # Aggregate candidates
    cands: List[Tuple[str, str, int, str]] = []  # (name, url, score, source)
    for name, href, s in readme_body_candidates(repo_id):
        cands.append((name, href, s, "readme_link"))
    for href, s in hydration_url_candidates(repo_url):
        cands.append(("", href, s, "hydrate_link"))
    for name, href, s in sibling_terms_candidates(repo_id):
        cands.append((name, href, s, "terms_file"))

    if not cands:
        return OtherOutcome(None, None, None, None, 0, "no-candidates")

    # Top candidate
    name, href, score, source = sorted(cands, key=lambda x: x[2], reverse=True)[0]
    vendor = _map_vendor_slug(href)
    fam = "proprietary" if vendor or score >= 40 else None

    # Try to derive a readable name from link text or filename
    derived_name = name or Path(href.split("?")[0].split("#")[0]).name.replace("-", " ")
    derived_name = derived_name.strip() if derived_name else None

    return OtherOutcome(
        license_slug=vendor,
        license_name=derived_name,
        license_url=href,
        license_family=fam,
        license_confidence=min(100, score),
        note=source,
    )


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
    reasons = "; ".join(r for r in [note, note2, note3, note4, note5] if r) or "unknown"
    return Outcome(None, reasons)


def enrich(records: List[Dict], interval: int = 100, verbose: bool = True
           ) -> Tuple[List[Dict], List[str]]:
    """
    For each record, if 'license' is missing/empty, attempt to resolve it.
    Additionally, for records with license == "other", attempt to enrich it
    with concrete proprietary metadata (and possibly promote the slug).
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

        current = (rec.get("license") or "").strip() if rec.get("license") is not None else ""
        updated = dict(rec)

        if not current:
            oc = resolve_license(repo_id, url)
            if oc.license:
                updated["license"] = oc.license
                updated["_license_source"] = oc.note
            else:
                misses.append(f"[{i}] {repo_id} ({oc.note})")
                if verbose:
                    print(f"[{i}] {repo_id}: MISS -> {oc.note}")
        else:
            # already has a license — leave as-is
            updated["_license_source"] = updated.get("_license_source") or "input"

        # SPECIAL: try to refine "other"
        if (updated.get("license") or "").strip().lower() == "other":
            other = resolve_other_license(repo_id, url)
            if other.license_name:
                updated["license_name"] = other.license_name
            if other.license_url:
                updated["license_url"] = other.license_url
            if other.license_family:
                updated["license_family"] = other.license_family
            updated["license_confidence"] = other.license_confidence
            updated["_license_source_other"] = other.note

            # If we have a strong vendor mapping, promote the slug
            if other.license_slug:
                updated["license_raw"] = "other"
                updated["license"] = other.license_slug
                updated["_license_source"] = f"{updated.get('_license_source','')};other-promoted".strip(";")

            if verbose and i % max(1, interval) == 0:
                print(f"[{i}] {repo_id}: OTHER ⇒ "
                      f"{updated.get('license')} "
                      f"({other.license_name or ''} @ {other.license_url or ''}; score {other.license_confidence})")

        out.append(updated)

        # checkpoint
        if (i + 1) % interval == 0:
            yield out, misses, i + 1, time.time() - t0

    yield out, misses, len(records), time.time() - t0


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Enrich HF licenses with robust fallbacks (+ 'other' resolver).")
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
    print("Fallbacks: API + static HTML + hydration + README front-matter + LICENSE fingerprint")
    print("PLUS: 'other' resolver (README links + hydration URLs + terms files)")

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
