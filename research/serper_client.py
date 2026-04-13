"""
Serper API client: company office research.

Provides:
  search(query, num) → list[dict]          — organic Serper results
  fetch_page(url)    → str | None          — fetched + extracted readable text

Loads SERPER_API_KEY from .env at project root (not from os.environ directly).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load .env from project root (research/serper_client.py → research/ → root)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH, override=True)

SERPER_ENDPOINT = "https://google.serper.dev/search"

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def _serper_key() -> str:
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        raise RuntimeError("SERPER_API_KEY not found in .env — add it before running the pipeline")
    return key


def search(query: str, num: int = 10) -> list[dict]:
    """
    Search via Serper and return organic results.
    Each result dict has: title, link, snippet, position.
    """
    try:
        resp = requests.post(
            SERPER_ENDPOINT,
            headers={"X-API-KEY": _serper_key(), "Content-Type": "application/json"},
            json={"q": query, "num": num},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("organic", [])
    except RuntimeError:
        raise
    except Exception as e:
        print(f"  Serper error for '{query}': {e}", file=sys.stderr)
        return []


def fetch_page(url: str, timeout: int = 20) -> Optional[str]:
    """
    Fetch a web page and return extracted readable text.
    Returns None on HTTP error or if content is too short.
    """
    try:
        resp = requests.get(
            url, headers=_HTTP_HEADERS, timeout=timeout, allow_redirects=True
        )
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} for {url}", file=sys.stderr)
            return None
        return _extract_text(resp.text, url)
    except Exception as e:
        print(f"  Fetch error for {url}: {e}", file=sys.stderr)
        return None


def _extract_text(raw: str, url: str) -> Optional[str]:
    """
    Extract readable text from raw HTML or plain-text content.
    For plain-text pages (low tag density, .txt, .md), normalize whitespace and return.
    For HTML, strip boilerplate and prefer main content containers.
    Mirrors the pattern in huggingface/fetch_license_texts.py:extract_text().
    """
    stripped = raw.strip()
    tag_count = stripped.count("<")

    if tag_count < 5 or url.endswith((".txt", ".md")):
        lines = [line.rstrip() for line in stripped.splitlines()]
        return "\n".join(lines).strip() or None

    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_=lambda c: c and any(
            x in c for x in ("content", "body", "main", "article", "post", "entry")
        ))
        or soup.find("body")
    )
    target = main if main else soup
    text = target.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    result = "\n".join(lines).strip()
    return result if len(result) > 100 else None
