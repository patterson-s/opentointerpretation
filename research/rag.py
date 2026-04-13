"""
Two-prompt RAG pipeline for AI company office location research.

Prompt 1 — extract_locations():
  Retrieves stored chunks for a company, reranks them, then asks
  Cohere Command-A to extract all HQ and affiliate office locations as JSON.

Prompt 2 — fact_check():
  For a single candidate location, retrieves and reranks relevant chunks,
  then asks Command-A to confirm or deny the claim with evidence.

Both prompts use cosine similarity (numpy, no pgvector required) for retrieval
and Cohere rerank-v4.0-pro before passing context to the model.
"""

from __future__ import annotations

import json
import sys
from typing import Optional

import numpy as np
import psycopg2
import psycopg2.extras

from research.cohere_client import embed_query, rerank, _client

COMMAND_MODEL = "command-a-03-2025"
RETRIEVE_TOP_N = 20   # chunks retrieved by cosine similarity before reranking
RERANK_TOP_N = 8      # chunks passed to Command-A after reranking


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _cosine_sim(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _fetch_company_chunks(conn, company_id: int) -> list[dict]:
    """Load all chunks (with embeddings) for a company's sources."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT rc.id, rc.chunk_text, rc.embedding, rs.url, rs.title
            FROM   research_chunks  rc
            JOIN   research_sources rs ON rc.source_id = rs.id
            WHERE  rs.company_id = %s
              AND  rc.embedding IS NOT NULL
            ORDER  BY rc.source_id, rc.chunk_index
            """,
            (company_id,),
        )
        return [dict(row) for row in cur.fetchall()]


def _retrieve_chunks(conn, company_id: int, query: str, top_n: int = RETRIEVE_TOP_N) -> list[dict]:
    """
    Retrieve top-N chunks for a company by cosine similarity to the query embedding.
    Falls back gracefully if no chunks have embeddings.
    """
    all_chunks = _fetch_company_chunks(conn, company_id)
    if not all_chunks:
        return []

    q_vec = embed_query(query)
    scored = [
        (chunk, _cosine_sim(q_vec, list(chunk["embedding"])))
        for chunk in all_chunks
        if chunk.get("embedding")
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scored[:top_n]]


# ── Prompt 1: Extract locations ───────────────────────────────────────────────

def extract_locations(conn, company_id: int, company_name: str) -> list[dict]:
    """
    Retrieve and rerank chunks for company_name, then prompt Command-A to
    extract all known HQ and affiliate office locations as structured JSON.

    Returns:
        list of {"finding_type": "headquarters"|"affiliate", "city": str, "country": str}
    """
    query = f"headquarters and office locations of {company_name}"
    top_chunks = _retrieve_chunks(conn, company_id, query, top_n=RETRIEVE_TOP_N)

    if not top_chunks:
        print(f"  [RAG] No embedded chunks found for {company_name}", file=sys.stderr)
        return []

    texts = [c["chunk_text"] for c in top_chunks]
    reranked = rerank(query, texts, top_n=RERANK_TOP_N)
    context = "\n\n---\n\n".join(r["text"] for r in reranked)

    co = _client()
    prompt = (
        f"Based on the sources below, identify ALL known office locations for {company_name}. "
        "Include the headquarters and every affiliate, regional, or satellite office. "
        "For each location specify the city and country. "
        "Return a JSON object with a single key 'locations', "
        "whose value is an array of objects with keys: "
        "'finding_type' (either 'headquarters' or 'affiliate'), 'city', 'country'.\n\n"
        f"Sources:\n{context}"
    )

    response = co.chat(
        model=COMMAND_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    try:
        result = json.loads(response.message.content[0].text)
        locations = result.get("locations", [])
        # Normalise finding_type to known values
        for loc in locations:
            if loc.get("finding_type") not in ("headquarters", "affiliate"):
                loc["finding_type"] = "affiliate"
        return locations
    except (json.JSONDecodeError, IndexError, KeyError) as exc:
        print(f"  [RAG] extract_locations parse error: {exc}", file=sys.stderr)
        return []


# ── Prompt 2: Fact-check ──────────────────────────────────────────────────────

def fact_check(
    conn,
    company_id: int,
    company_name: str,
    candidate: dict,
) -> tuple[bool, str]:
    """
    Verify a candidate location against the company's stored chunks.

    Args:
        candidate: {"finding_type": str, "city": str, "country": str}

    Returns:
        (confirmed: bool, notes: str)
    """
    city = candidate.get("city", "")
    country = candidate.get("country", "")
    finding_type = candidate.get("finding_type", "office")

    query = f"{company_name} {finding_type} in {city} {country}"
    top_chunks = _retrieve_chunks(conn, company_id, query, top_n=RETRIEVE_TOP_N)

    if not top_chunks:
        return False, "No supporting chunks found in stored sources"

    texts = [c["chunk_text"] for c in top_chunks]
    reranked = rerank(query, texts, top_n=RERANK_TOP_N)
    context = "\n\n---\n\n".join(r["text"] for r in reranked)

    co = _client()
    prompt = (
        f"Based only on the sources below, determine whether {company_name} has "
        f"its {finding_type} located in {city}, {country}. "
        "Return a JSON object with two keys: "
        "'confirmed' (boolean — true if the sources support this claim), "
        "'notes' (a brief explanation citing specific evidence from the sources).\n\n"
        f"Sources:\n{context}"
    )

    response = co.chat(
        model=COMMAND_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    try:
        result = json.loads(response.message.content[0].text)
        return bool(result.get("confirmed", False)), result.get("notes", "")
    except (json.JSONDecodeError, IndexError, KeyError) as exc:
        print(f"  [RAG] fact_check parse error: {exc}", file=sys.stderr)
        return False, f"Parse error: {exc}"
