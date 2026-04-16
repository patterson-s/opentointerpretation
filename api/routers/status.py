"""
Status router — exposes last collection run info.

GET /api/status → {last_collected_at, new_models_added, total_models_seen,
                    total_models_in_db, status}
"""

from __future__ import annotations

from fastapi import APIRouter

from api.db import get_cursor

router = APIRouter(prefix="/api/status", tags=["status"])


@router.get("")
def get_status() -> dict:
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT started_at, completed_at, status,
                   new_models_added, total_models_seen
            FROM collection_runs
            ORDER BY started_at DESC
            LIMIT 1
            """
        )
        run = cur.fetchone()

        cur.execute(
            "SELECT COUNT(*) AS total FROM models WHERE data_source = 'huggingface'"
        )
        total = cur.fetchone()["total"]

    if run is None:
        return {
            "last_collected_at": None,
            "new_models_added": None,
            "total_models_seen": None,
            "total_models_in_db": total,
            "status": "never_run",
        }

    return {
        "last_collected_at": run["completed_at"] or run["started_at"],
        "new_models_added": run["new_models_added"],
        "total_models_seen": run["total_models_seen"],
        "total_models_in_db": total,
        "status": run["status"],
    }
