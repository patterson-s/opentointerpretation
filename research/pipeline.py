"""
AI Company Office Research Pipeline.

Searches for headquarters and affiliate offices of frontier AI companies,
fetches and stores sources with full provenance, chunks + embeds them via Cohere,
then uses a two-prompt RAG system to extract and fact-check locations.

Usage:
    python research/pipeline.py                         # all companies
    python research/pipeline.py --companies "OpenAI"    # single company
    python research/pipeline.py --max-sources 5 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Force UTF-8 output on Windows to avoid cp1252 errors with Unicode in LLM responses
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on sys.path so `research` package is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env from project root before any submodule imports that need env vars
_ENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(_ENV_PATH, override=True)

from research import serper_client, cohere_client, rag  # noqa: E402 (after load_dotenv)


# ── Company search queries ────────────────────────────────────────────────────
# Tuples of (display_name_in_db, [search_query_1, search_query_2])
# display_name must match companies.display_name exactly for the DB lookup.
# NOTE: Cohere HQ is Toronto, CA — California office is an affiliate.

COMPANIES: list[tuple[str, list[str]]] = [
    (
        "OpenAI",
        [
            "OpenAI headquarters location city country",
            "OpenAI offices worldwide locations list",
        ],
    ),
    (
        "Anthropic",
        [
            "Anthropic AI headquarters location city country",
            "Anthropic offices worldwide locations",
        ],
    ),
    (
        "Google DeepMind",
        [
            "Google DeepMind headquarters location city country",
            "Google DeepMind offices worldwide locations",
        ],
    ),
    (
        "xAI",
        [
            "xAI Elon Musk AI company headquarters location",
            "xAI artificial intelligence offices locations",
        ],
    ),
    (
        "Meta AI (FAIR)",
        [
            "Meta AI headquarters location city country",
            "Meta FAIR research offices worldwide locations",
        ],
    ),
    (
        "Cohere",
        [
            "Cohere AI headquarters Toronto Canada location",
            "Cohere AI offices worldwide locations",
        ],
    ),
    (
        "Mistral AI",
        [
            "Mistral AI headquarters location Paris France",
            "Mistral AI offices worldwide locations",
        ],
    ),
    (
        "Baidu",
        [
            "Baidu AI headquarters location city country",
            "Baidu research offices worldwide locations",
        ],
    ),
    (
        "DeepSeek",
        [
            "DeepSeek AI headquarters location Hangzhou China",
            "DeepSeek offices locations",
        ],
    ),
    (
        "Qwen (Alibaba DAMO)",
        [
            "Alibaba DAMO Academy headquarters location Hangzhou China",
            "Alibaba AI Qwen offices worldwide locations",
        ],
    ),
]


# ── DB connection ─────────────────────────────────────────────────────────────

def _connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ.get("PGPORT", 5432)),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ.get("PGPASSWORD", ""),
    )


def _get_company_id(conn, display_name: str) -> Optional[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM companies WHERE display_name = %s", (display_name,))
        row = cur.fetchone()
        return row[0] if row else None


def _upsert_source(
    conn,
    company_id: int,
    url: str,
    title: Optional[str],
    snippet: Optional[str],
    raw_content: Optional[str],
    search_query: str,
    source_type: str = "serper_result",
) -> int:
    """
    Insert a new source or update its content if already present.
    Returns the source id.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO research_sources
                (company_id, url, title, snippet, raw_content, search_query, source_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE SET
                raw_content = COALESCE(EXCLUDED.raw_content, research_sources.raw_content),
                company_id  = COALESCE(research_sources.company_id, EXCLUDED.company_id)
            RETURNING id
            """,
            (company_id, url, title, snippet, raw_content, search_query, source_type),
        )
        return cur.fetchone()[0]


def _save_chunks(
    conn,
    source_id: int,
    chunks: list[str],
    embeddings: list[list[float]],
) -> None:
    with conn.cursor() as cur:
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """
                INSERT INTO research_chunks (source_id, chunk_index, chunk_text, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source_id, chunk_index) DO NOTHING
                """,
                (source_id, idx, chunk_text, embedding),
            )


def _save_finding(
    conn,
    company_id: int,
    finding_type: str,
    city: str,
    country: str,
    confidence: str,
    source_count: int,
    notes: str,
    source_ids: list[int],
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO research_findings
                (company_id, finding_type, city, country, confidence, source_count, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (company_id, finding_type, city, country) DO UPDATE SET
                confidence   = EXCLUDED.confidence,
                source_count = EXCLUDED.source_count,
                notes        = EXCLUDED.notes,
                extracted_at = NOW()
            RETURNING id
            """,
            (company_id, finding_type, city, country, confidence, source_count, notes),
        )
        row = cur.fetchone()
        if row:
            finding_id = row[0]
            for src_id in source_ids:
                cur.execute(
                    """
                    INSERT INTO research_finding_sources (finding_id, source_id)
                    VALUES (%s, %s) ON CONFLICT DO NOTHING
                    """,
                    (finding_id, src_id),
                )


# ── Phase 1: Search, fetch, embed ────────────────────────────────────────────

def process_sources(
    conn,
    company_id: int,
    company_name: str,
    queries: list[str],
    max_sources: int,
    dry_run: bool,
) -> list[int]:
    """
    For each search query:
      1. Run Serper search
      2. Fetch up to max_sources pages
      3. Chunk and embed content
      4. Store in research_sources + research_chunks

    Returns list of source_ids created/updated.
    """
    source_ids: list[int] = []
    seen_urls: set[str] = set()

    for query in queries:
        print(f"\n  Searching: {query!r}")
        results = serper_client.search(query, num=10)
        time.sleep(0.5)  # polite rate-limit

        fetched = 0
        for result in results:
            if fetched >= max_sources:
                break
            url = result.get("link", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title = result.get("title")
            snippet = result.get("snippet")

            print(f"    [{fetched + 1}/{max_sources}] {url[:90]}")
            content = serper_client.fetch_page(url)
            time.sleep(0.3)

            if dry_run:
                n_chunks = len(cohere_client.chunk_text(content)) if content else 0
                print(f"    [DRY-RUN] title={title!r}, content={'yes' if content else 'no'} ({n_chunks} chunks)")
                fetched += 1
                continue

            source_id = _upsert_source(
                conn, company_id, url, title, snippet, content, query
            )
            conn.commit()
            source_ids.append(source_id)

            if content:
                chunks = cohere_client.chunk_text(content)
                if chunks:
                    print(f"    Embedding {len(chunks)} chunk(s)...")
                    embeddings = cohere_client.embed_texts(chunks)
                    _save_chunks(conn, source_id, chunks, embeddings)
                    conn.commit()
                    time.sleep(0.3)

            fetched += 1

    return source_ids


# ── Phase 3 helper: substantiate with extra search ────────────────────────────

def _substantiate(
    conn,
    company_id: int,
    company_name: str,
    candidate: dict,
    max_extra: int = 3,
) -> list[int]:
    """
    Search for up to max_extra additional sources that might substantiate
    a candidate location. Fetches, chunks, and embeds them.
    Returns list of new source_ids.
    """
    city = candidate.get("city", "")
    country = candidate.get("country", "")
    finding_type = candidate.get("finding_type", "office")
    query = f"{company_name} {finding_type} {city} {country}"

    print(f"    Substantiating via: {query!r}")
    results = serper_client.search(query, num=5)
    time.sleep(0.5)

    extra_ids: list[int] = []
    for result in results[:max_extra]:
        url = result.get("link", "")
        if not url:
            continue
        content = serper_client.fetch_page(url)
        time.sleep(0.3)
        src_id = _upsert_source(
            conn,
            company_id,
            url,
            result.get("title"),
            result.get("snippet"),
            content,
            query,
        )
        conn.commit()
        extra_ids.append(src_id)

        if content:
            chunks = cohere_client.chunk_text(content)
            if chunks:
                embeddings = cohere_client.embed_texts(chunks)
                _save_chunks(conn, src_id, chunks, embeddings)
                conn.commit()
                time.sleep(0.3)

    return extra_ids


# ── Per-company orchestration ─────────────────────────────────────────────────

def run_company(
    conn,
    company_name: str,
    queries: list[str],
    max_sources: int,
    dry_run: bool,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"  COMPANY: {company_name}")
    print(f"{'=' * 60}")

    company_id = _get_company_id(conn, company_name)
    if company_id is None:
        print(f"  WARNING: '{company_name}' not found in companies table — skipping.")
        print(f"  Tip: check that the display_name matches exactly.")
        return

    # ── Phase 1: Search & embed ──────────────────────────────────────────────
    print("\n[Phase 1] Searching and embedding sources...")
    source_ids = process_sources(
        conn, company_id, company_name, queries, max_sources, dry_run
    )
    print(f"  Stored {len(source_ids)} source(s).")

    if dry_run:
        print(f"\n[DRY-RUN] Skipping RAG phases for {company_name}.")
        return

    # ── Phase 2: Extract candidate locations (Prompt 1) ──────────────────────
    print(f"\n[Phase 2] Extracting locations via RAG (Prompt 1)...")
    candidates = rag.extract_locations(conn, company_id, company_name)
    if not candidates:
        print("  No candidates extracted.")
        return

    print(f"  Extracted {len(candidates)} candidate location(s):")
    for c in candidates:
        print(f"    {c.get('finding_type','?'):14s} {c.get('city','?')}, {c.get('country','?')}")

    # ── Phase 3: Fact-check each candidate (Prompt 2) ────────────────────────
    print(f"\n[Phase 3] Fact-checking locations (Prompt 2)...")
    for candidate in candidates:
        city = candidate.get("city", "").strip()
        country = candidate.get("country", "").strip()
        finding_type = candidate.get("finding_type", "affiliate")
        if not city or not country:
            continue

        print(f"\n  Checking: {finding_type} -> {city}, {country}")

        # Fetch additional sources to help substantiate (up to 3)
        extra_ids = _substantiate(conn, company_id, company_name, candidate, max_extra=3)

        # Run fact-check prompt against all available chunks
        confirmed, notes = rag.fact_check(conn, company_id, company_name, candidate)
        confidence = "confirmed" if confirmed else "unconfirmed"
        all_src_ids = source_ids + extra_ids

        print(f"  Result: {confidence.upper()} ({len(all_src_ids)} sources)")
        print(f"  Notes:  {notes[:120]}")

        _save_finding(
            conn,
            company_id,
            finding_type,
            city,
            country,
            confidence,
            len(all_src_ids),
            notes,
            all_src_ids,
        )
        conn.commit()
        time.sleep(0.3)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research AI company headquarters and affiliate office locations."
    )
    parser.add_argument(
        "--companies",
        nargs="+",
        metavar="NAME",
        help="Company display names to research (default: all). "
             "Wrap multi-word names in quotes, e.g. --companies 'OpenAI' 'Meta AI'",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=5,
        metavar="N",
        help="Maximum pages to fetch per search query (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run searches and print results without writing to DB or calling RAG",
    )
    args = parser.parse_args()

    # Validate required environment variables
    missing = [k for k in ("SERPER_API_KEY", "COHERE_API_KEY") if not os.environ.get(k)]
    if missing:
        for k in missing:
            print(f"ERROR: {k} not found in .env — add it to run the pipeline.", file=sys.stderr)
        sys.exit(1)

    # Filter companies if --companies specified
    target = COMPANIES
    if args.companies:
        lowered = {n.lower() for n in args.companies}
        target = [(n, q) for n, q in COMPANIES if n.lower() in lowered]
        if not target:
            print(
                f"No matching companies. Available names:\n  "
                + "\n  ".join(n for n, _ in COMPANIES),
                file=sys.stderr,
            )
            sys.exit(1)

    conn = _connect()
    try:
        for company_name, queries in target:
            run_company(conn, company_name, queries, args.max_sources, args.dry_run)
    finally:
        conn.close()

    print("\n\nPipeline complete.")


if __name__ == "__main__":
    main()
