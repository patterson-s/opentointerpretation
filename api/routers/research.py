"""
Research API endpoints.

GET  /api/research/map                              — all geocoded findings for map visualization
GET  /api/research/companies/{company_id}/sources   — sources fetched for a company
GET  /api/research/companies/{company_id}/findings  — extracted + fact-checked locations
GET  /api/research/sources/{source_id}/chunks       — text chunks for a source
POST /api/research/query                            — ad-hoc RAG query over company chunks
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.db import get_cursor

router = APIRouter(prefix="/api/research", tags=["research"])

# Ensure .env is loaded for Cohere key when this module is imported by uvicorn
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_ENV_PATH, override=True)


# ── Map endpoint ─────────────────────────────────────────────────────────────

@router.get("/map")
def get_map_data() -> list[dict]:
    """
    Return all geocoded research findings for map visualization.
    Each row includes company_id, company_name, finding_type, city, country,
    confidence, latitude, and longitude.
    Only rows with coordinates are returned (skips 'Various' regional placeholders).
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                c.id          AS company_id,
                c.display_name AS company_name,
                f.finding_type,
                f.city,
                f.country,
                f.confidence,
                f.latitude,
                f.longitude
            FROM   research_findings f
            JOIN   companies c ON f.company_id = c.id
            WHERE  f.latitude  IS NOT NULL
              AND  f.longitude IS NOT NULL
            ORDER  BY c.display_name, f.finding_type DESC, f.city
            """
        )
        return [dict(row) for row in cur.fetchall()]


# ── Source endpoints ──────────────────────────────────────────────────────────

@router.get("/companies/{company_id}/sources")
def list_company_sources(company_id: int) -> list[dict]:
    """List all research sources fetched for a company."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, url, title, snippet, search_query, source_type, fetched_at
            FROM   research_sources
            WHERE  company_id = %s
            ORDER  BY fetched_at DESC
            """,
            (company_id,),
        )
        return [dict(row) for row in cur.fetchall()]


@router.get("/companies/{company_id}/findings")
def list_company_findings(company_id: int) -> list[dict]:
    """List all extracted location findings for a company, with confidence."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, finding_type, city, country, confidence, source_count, notes, extracted_at
            FROM   research_findings
            WHERE  company_id = %s
            ORDER  BY finding_type, confidence DESC, city
            """,
            (company_id,),
        )
        return [dict(row) for row in cur.fetchall()]


@router.get("/sources/{source_id}/chunks")
def list_source_chunks(source_id: int) -> list[dict]:
    """List text chunks for a given source (embeddings excluded from response)."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, chunk_index, chunk_text
            FROM   research_chunks
            WHERE  source_id = %s
            ORDER  BY chunk_index
            """,
            (source_id,),
        )
        return [dict(row) for row in cur.fetchall()]


# ── RAG query endpoint ────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    company_id: int
    question: str


@router.post("/query")
def rag_query(body: QueryRequest) -> dict:
    """
    Ad-hoc RAG query over a company's stored chunks.
    Retrieves chunks by cosine similarity, reranks with Cohere, then
    answers the question using Cohere Command-A.
    """
    from research.cohere_client import embed_query, rerank, _client

    RETRIEVE_TOP_N = 20
    RERANK_TOP_N = 8

    # Load chunks with embeddings for this company
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT rc.chunk_text, rc.embedding
            FROM   research_chunks  rc
            JOIN   research_sources rs ON rc.source_id = rs.id
            WHERE  rs.company_id = %s
              AND  rc.embedding IS NOT NULL
            ORDER  BY rs.id, rc.chunk_index
            """,
            (body.company_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No embedded chunks found for company_id={body.company_id}. "
                   "Run the research pipeline first.",
        )

    # Cosine similarity retrieval
    def _cosine(a, b):
        va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

    q_vec = embed_query(body.question)
    scored = sorted(
        [(r["chunk_text"], _cosine(q_vec, list(r["embedding"]))) for r in rows],
        key=lambda x: x[1],
        reverse=True,
    )[:RETRIEVE_TOP_N]

    texts = [t for t, _ in scored]
    reranked = rerank(body.question, texts, top_n=RERANK_TOP_N)
    context = "\n\n---\n\n".join(r["text"] for r in reranked)

    co = _client()
    response = co.chat(
        model="command-a-03-2025",
        messages=[
            {
                "role": "user",
                "content": (
                    f"{body.question}\n\n"
                    f"Answer based only on the following sources:\n\n{context}"
                ),
            }
        ],
        temperature=0.3,
    )

    return {"answer": response.message.content[0].text}
