#!/usr/bin/env python3
"""
Fetch license texts for all unique license slugs in the database.

Sources (tried in order per slug):
  1. CANONICAL_URLS dict  — direct HTTP GET to known URL
  2. Serper search         — Google search via Serper API, then fetch first useful result

Updates the `licenses` table with `license_text` and `source_url`.

Usage:
    python huggingface/fetch_license_texts.py             # fetch all missing
    python huggingface/fetch_license_texts.py --dry-run   # print what would be fetched
    python huggingface/fetch_license_texts.py --slug mit  # single slug
    python huggingface/fetch_license_texts.py --force     # re-fetch all (even if text already present)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import warnings

import psycopg2
import psycopg2.extras
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Load .env from project root (two levels up from this file)
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# ── Canonical URL map ─────────────────────────────────────────────────────────
# Maps license slug → URL of the license text (plain text or HTML).
# HTML pages will be scraped for main content; plain text returned as-is.

CANONICAL_URLS: dict[str, str] = {
    # Standard SPDX / OSS licenses
    "apache-2.0":             "https://www.apache.org/licenses/LICENSE-2.0.txt",
    "mit":                    "https://spdx.org/licenses/MIT.html",
    "ms-pl":                  "https://spdx.org/licenses/MS-PL.html",
    "cc-by-4.0":              "https://creativecommons.org/licenses/by/4.0/legalcode.txt",
    "cc-by-nc-4.0":           "https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt",
    "cc-by-nc-sa-4.0":        "https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt",
    "cc-by-nc-2.0":           "https://creativecommons.org/licenses/by-nc/2.0/legalcode",
    "cc-by-sa-4.0":           "https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt",
    "cdla-permissive-2.0":    "https://cdla.dev/permissive-2-0/",
    # Meta Llama (GitHub raw — llama.com blocks bots)
    "llama2":    "https://raw.githubusercontent.com/meta-llama/llama/main/LICENSE",
    "llama3":    "https://raw.githubusercontent.com/meta-llama/llama3/main/LICENSE",
    "llama3.1":  "https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE",
    "llama3.2":  "https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_2/LICENSE",
    "llama3.3":  "https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_3/LICENSE",
    # Google Gemma
    "gemma":     "https://ai.google.dev/gemma/terms",
    # HuggingFace-hosted raw files (use public/non-gated repos)
    "bigscience-bloom-rail-1.0":
        "https://huggingface.co/bigscience/bloom",
    "creativeml-openrail-m":
        "https://huggingface.co/blog/open_rail",
    "openrail++":
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/LICENSE.md",
    "jamba-open-model-license":
        "https://www.ai21.com/licenses/jamba-open-model-license",
    "tongyi-qianwen":          "https://huggingface.co/Qwen/Qwen-7B/raw/main/LICENSE",
    "tongyi-qianwen-research": "https://huggingface.co/Qwen/Qwen1.5-0.5B/raw/main/LICENSE",
    "qwen":                    "https://huggingface.co/Qwen/Qwen2-7B/raw/main/LICENSE",
    "qwen-research":           "https://huggingface.co/Qwen/Qwen2.5-3B/raw/main/LICENSE",
    # NVIDIA licenses (public HF repos or developer.nvidia.com)
    "nvclv1":                  "https://huggingface.co/nvidia/MambaVision-S-1K/raw/main/LICENSE",
    "nvlicense":               "https://huggingface.co/nvidia/VideoITG-8B/raw/main/LICENSE",
    "nvidia-source-code-license":
        "https://huggingface.co/nvidia/RADIO/raw/main/LICENSE",
    "nvidia-oneway-noncommercial-license":
        "https://developer.nvidia.com/downloads/assets/secure/licensing/nvidia-oneway-noncommercial-license-agreement.pdf",
    "nvidia-internal-scientific-research-and-development-model-license":
        "https://raw.githubusercontent.com/NVIDIA/NeMo/main/LICENSE",
    "nvidia-open-model-license":
        "https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf",
}

# Slugs to skip (too generic; no meaningful text to fetch)
SKIP_SLUGS: set[str] = {"other", "unknown"}

# Serper search hints for slugs not in CANONICAL_URLS
SERPER_SEARCH_HINTS: dict[str, str] = {
    "hybrid_MIT_metallama3.1commnuity":
        "Meta Llama 3.1 community hybrid MIT license agreement full text",
    "hybrid_mit_qwen":
        "Qwen hybrid MIT license agreement full text",
}


# ── DB connection ──────────────────────────────────────────────────────────────

def get_conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ.get("PGPORT", 5432)),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ.get("PGPASSWORD", ""),
    )


# ── HTTP helpers ──────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def fetch_url(url: str, timeout: int = 20) -> Optional[str]:
    """Fetch a URL and return raw text content, or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if resp.status_code == 200:
            return resp.text
        print(f"  HTTP {resp.status_code} for {url}", file=sys.stderr)
    except Exception as e:
        print(f"  Fetch error for {url}: {e}", file=sys.stderr)
    return None


def extract_text(raw: str, url: str) -> Optional[str]:
    """
    Extract meaningful text from raw HTML (or plain text).
    For plain-text URLs (LICENSE files, .txt, .md), return as-is after stripping.
    For HTML pages, strip markup and prefer main content areas.
    """
    # Plain text heuristic: very low tag density
    stripped = raw.strip()
    tag_count = stripped.count("<")
    if tag_count < 5 or url.endswith((".txt", ".md", "/LICENSE", "/license")):
        # Likely plain text — just normalize whitespace
        lines = [l.rstrip() for l in stripped.splitlines()]
        return "\n".join(lines).strip() or None

    # HTML: parse and extract main content
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    # Prefer semantic content containers
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_=lambda c: c and any(
            x in c for x in ("license", "content", "body", "terms", "legal")
        ))
        or soup.find("body")
    )
    target = main if main else soup
    text = target.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    result = "\n".join(lines).strip()
    return result if len(result) > 100 else None


# ── Serper search ─────────────────────────────────────────────────────────────

def serper_search(query: str, num: int = 5) -> list[dict]:
    """Return top organic search results from Serper."""
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        print("  SERPER_API_KEY not set; skipping Serper search", file=sys.stderr)
        return []
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": query, "num": num},
            timeout=15,
        )
        return resp.json().get("organic", [])
    except Exception as e:
        print(f"  Serper error: {e}", file=sys.stderr)
        return []


def fetch_via_serper(query: str) -> tuple[str, str] | None:
    """
    Search for a license using Serper, then fetch text from the first useful result.
    Returns (url, text) or None.
    """
    results = serper_search(query)
    for result in results:
        url = result.get("link", "")
        if not url:
            continue
        raw = fetch_url(url)
        if raw:
            text = extract_text(raw, url)
            if text and len(text) > 200:
                return url, text
        time.sleep(0.3)
    return None


# ── Main fetcher ──────────────────────────────────────────────────────────────

class LicenseFetcher:
    def __init__(self, dry_run: bool = False, force: bool = False):
        self.dry_run = dry_run
        self.force = force

    def load_pending_slugs(self) -> list[str]:
        """Return slugs that don't yet have license_text (or all if --force)."""
        conn = get_conn()
        with conn, conn.cursor() as cur:
            if self.force:
                cur.execute("SELECT slug FROM licenses ORDER BY slug")
            else:
                cur.execute(
                    "SELECT slug FROM licenses WHERE license_text IS NULL ORDER BY slug"
                )
            return [row[0] for row in cur.fetchall()]

    def update_db(self, slug: str, text: str, source_url: str) -> None:
        if self.dry_run:
            print(f"  [DRY-RUN] Would set license_text ({len(text)} chars) + source_url={source_url}")
            return
        conn = get_conn()
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE licenses
                SET license_text = %s,
                    source_url   = %s
                WHERE slug = %s
                """,
                (text, source_url, slug),
            )
        print(f"  Saved {len(text)} chars, source: {source_url}")

    def process_slug(self, slug: str) -> None:
        print(f"\n[{slug}]", file=sys.stderr)

        if slug in SKIP_SLUGS:
            print(f"  Skipping ({slug} is generic)", file=sys.stderr)
            return

        # 1. Canonical URL
        if slug in CANONICAL_URLS:
            url = CANONICAL_URLS[slug]
            print(f"  Trying canonical URL: {url}", file=sys.stderr)
            raw = fetch_url(url)
            if raw:
                text = extract_text(raw, url)
                if text:
                    self.update_db(slug, text, url)
                    return
                print(f"  Text extraction failed for {url}", file=sys.stderr)

        # 2. Serper hint
        if slug in SERPER_SEARCH_HINTS:
            query = SERPER_SEARCH_HINTS[slug]
            print(f"  Trying Serper hint: {query!r}", file=sys.stderr)
            result = fetch_via_serper(query)
            if result:
                url, text = result
                self.update_db(slug, text, url)
                return

        # 3. Generic Serper fallback
        query = f'"{slug}" AI model license agreement full text site:huggingface.co OR site:github.com OR site:opensource.org'
        print(f"  Trying generic Serper: {query!r}", file=sys.stderr)
        result = fetch_via_serper(query)
        if result:
            url, text = result
            self.update_db(slug, text, url)
            return

        print(f"  MISS: no text found for {slug}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch license texts into the database")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing to DB")
    parser.add_argument("--slug", metavar="SLUG", help="Process a single license slug only")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if text already present")
    args = parser.parse_args()

    fetcher = LicenseFetcher(dry_run=args.dry_run, force=args.force)

    if args.slug:
        slugs = [args.slug]
    else:
        slugs = fetcher.load_pending_slugs()
        print(f"Found {len(slugs)} slugs to process.", file=sys.stderr)

    for i, slug in enumerate(slugs, 1):
        print(f"\n({i}/{len(slugs)}) Processing: {slug}", file=sys.stderr)
        fetcher.process_slug(slug)
        time.sleep(0.5)  # polite rate-limit

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
