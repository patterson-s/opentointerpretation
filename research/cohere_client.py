"""
Cohere client for the research pipeline.

Provides:
  chunk_text(text)           → list[str]           — sentence-boundary chunking
  embed_texts(texts)         → list[list[float]]   — embed-v4.0, 1024-dim, search_document
  embed_query(query)         → list[float]         — embed-v4.0, 1024-dim, search_query
  rerank(query, docs, top_n) → list[dict]          — rerank-v4.0-pro results

Loads COHERE_API_KEY from .env at project root (not from os.environ directly).
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import cohere
from dotenv import load_dotenv

# Load .env from project root
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH, override=True)

EMBED_MODEL = "embed-v4.0"
RERANK_MODEL = "rerank-v4.0-pro"
EMBED_DIM = 1024

# Chunking parameters (in characters; ~1500 chars ≈ 375 tokens for English)
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MIN_CHUNK_LEN = 50


def _client() -> cohere.ClientV2:
    key = os.environ.get("COHERE_API_KEY")
    if not key:
        raise RuntimeError("COHERE_API_KEY not found in .env file")
    return cohere.ClientV2(api_key=key)


def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks at sentence boundaries.
    Target CHUNK_SIZE characters per chunk, CHUNK_OVERLAP character overlap.
    Returns only chunks longer than MIN_CHUNK_LEN characters.
    """
    if not text or not text.strip():
        return []

    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= CHUNK_SIZE:
            current = (current + " " + sentence).strip() if current else sentence
        else:
            if current:
                chunks.append(current)
            # Begin next chunk with overlap tail from previous chunk
            if len(current) > CHUNK_OVERLAP:
                tail = current[-CHUNK_OVERLAP:].strip()
                # Trim to a sentence boundary within the tail if possible
                tail_split = re.split(r"(?<=[.!?])\s+", tail)
                tail = " ".join(tail_split[1:]).strip() if len(tail_split) > 1 else tail
                current = (tail + " " + sentence).strip()
            else:
                current = sentence

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c) >= MIN_CHUNK_LEN]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of document texts using Cohere embed-v4.0 (1024-dim).
    Processes in batches of 96 (Cohere max per call).
    """
    if not texts:
        return []

    co = _client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), 96):
        batch = texts[i : i + 96]
        response = co.embed(
            model=EMBED_MODEL,
            texts=batch,
            input_type="search_document",
            embedding_types=["float"],
            output_dimension=EMBED_DIM,
        )
        all_embeddings.extend(response.embeddings.float)
        if i + 96 < len(texts):
            time.sleep(0.3)

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """
    Embed a single search query using Cohere embed-v4.0 (1024-dim).
    Uses input_type='search_query' (vs 'search_document' for corpus chunks).
    """
    co = _client()
    response = co.embed(
        model=EMBED_MODEL,
        texts=[query],
        input_type="search_query",
        embedding_types=["float"],
        output_dimension=EMBED_DIM,
    )
    return response.embeddings.float[0]


def rerank(query: str, documents: list[str], top_n: int = 5) -> list[dict]:
    """
    Rerank documents against a query using Cohere rerank-v4.0-pro.
    Returns list of {"index": int, "relevance_score": float, "text": str}.
    """
    if not documents:
        return []

    co = _client()
    top_n = min(top_n, len(documents))
    response = co.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=top_n,
    )
    return [
        {
            "index": r.index,
            "relevance_score": r.relevance_score,
            "text": documents[r.index],
        }
        for r in response.results
    ]
