"""
Home router — weekly digest and collection status for the Home tab.

GET /api/home            → latest digest + last 12 weeks history + collection status
GET /api/home/digests/{id} → single digest by id
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.db import get_cursor

router = APIRouter(prefix="/api/home", tags=["home"])


@router.get("")
def get_home() -> dict:
    with get_cursor() as cur:
        # Latest digest
        cur.execute(
            """
            SELECT id, week_start, week_end, new_models,
                   by_company, by_license, by_modality, narrative, generated_at
            FROM weekly_digests
            ORDER BY week_start DESC
            LIMIT 1
            """
        )
        latest = cur.fetchone()

        # History — last 12 weeks (excluding the latest already fetched)
        cur.execute(
            """
            SELECT id, week_start, week_end, new_models,
                   by_company, by_license, by_modality, narrative, generated_at
            FROM weekly_digests
            ORDER BY week_start DESC
            LIMIT 12
            """
        )
        history = cur.fetchall()

        # Collection status (mirrors /api/status)
        cur.execute(
            """
            SELECT started_at, completed_at, status, new_models_added, total_models_seen
            FROM collection_runs
            ORDER BY started_at DESC
            LIMIT 1
            """
        )
        run = cur.fetchone()

        cur.execute("SELECT COUNT(*) AS total FROM models WHERE data_source = 'huggingface'")
        total_hf = cur.fetchone()["total"]

        cur.execute("SELECT COUNT(*) AS total FROM models")
        total_all = cur.fetchone()["total"]

    collection_status = {
        "last_collected_at": None,
        "new_models_added": None,
        "total_models_in_db": total_hf,
        "total_all_models": total_all,
        "status": "never_run",
    }
    if run:
        collection_status.update({
            "last_collected_at": run["completed_at"] or run["started_at"],
            "new_models_added": run["new_models_added"],
            "status": run["status"],
        })

    return {
        "latest_digest": dict(latest) if latest else None,
        "history": [dict(r) for r in history],
        "collection_status": collection_status,
    }


@router.get("/digests/{digest_id}")
def get_digest(digest_id: int) -> dict:
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, week_start, week_end, new_models,
                   by_company, by_license, by_modality, narrative, generated_at
            FROM weekly_digests
            WHERE id = %s
            """,
            (digest_id,),
        )
        row = cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Digest not found")
    return dict(row)
