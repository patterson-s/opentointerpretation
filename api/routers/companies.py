"""
/api/companies routes.

GET /api/companies        — list all companies (for selector)
GET /api/companies/{id}   — full detail for one company
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.db import get_cursor

router = APIRouter(prefix="/api/companies", tags=["companies"])


@router.get("")
def list_companies() -> list[dict]:
    """Return all companies sorted by display name."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, display_name, country_hq, hf_handle
            FROM companies
            ORDER BY display_name
            """
        )
        return [dict(row) for row in cur.fetchall()]


@router.get("/{company_id}")
def get_company(company_id: int) -> dict:
    """Return full detail for a single company."""
    with get_cursor() as cur:
        # Company info
        cur.execute(
            """
            SELECT id, display_name, country_hq, hf_handle
            FROM companies
            WHERE id = %s
            """,
            (company_id,),
        )
        company = cur.fetchone()
        if company is None:
            raise HTTPException(status_code=404, detail="Company not found")

        # Model count + data source breakdown
        cur.execute(
            """
            SELECT
                COUNT(*)                                             AS model_count,
                COUNT(*) FILTER (WHERE data_source = 'huggingface') AS hf_count,
                COUNT(*) FILTER (WHERE data_source != 'huggingface') AS closed_count
            FROM models
            WHERE company_id = %s
            """,
            (company_id,),
        )
        counts = cur.fetchone()

        # License distribution
        cur.execute(
            """
            SELECT
                COALESCE(l.slug, 'unknown') AS slug,
                COUNT(*)                    AS count
            FROM models m
            LEFT JOIN licenses l ON m.license_id = l.id
            WHERE m.company_id = %s
            GROUP BY l.slug
            ORDER BY count DESC
            """,
            (company_id,),
        )
        license_dist = [dict(row) for row in cur.fetchall()]

    result = dict(company)
    result["model_count"] = counts["model_count"]
    result["hf_count"] = counts["hf_count"]
    result["closed_count"] = counts["closed_count"]
    result["license_distribution"] = license_dist
    return result
