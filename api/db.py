"""
Database connection helpers for the opentointerpretation API.
Uses psycopg2 + python-dotenv, mirroring the pattern in db/ingest.py.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file: api/db.py → root)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH, override=True)


def _connect() -> psycopg2.extensions.connection:
    # Production: Neon (or any hosted Postgres) supplies a full DATABASE_URL.
    # Local dev: fall back to individual PG* env vars from .env.
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url)
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ.get("PGPORT", 5432)),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ.get("PGPASSWORD", ""),
    )


@contextmanager
def get_cursor() -> Generator[psycopg2.extensions.cursor, None, None]:
    """Context manager that yields a dict-mode cursor and auto-closes."""
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
        conn.commit()
    finally:
        conn.close()
