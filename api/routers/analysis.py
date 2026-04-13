"""
/api/analysis routes.

GET /api/analysis/model-releases        — model releases by company
GET /api/analysis/license-trends       — license distribution trends
GET /api/analysis/country-comparison    — country comparison data
GET /api/analysis/time-analysis         — time-based analysis data
"""

from __future__ import annotations

from fastapi import APIRouter

from api.db import get_cursor

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


@router.get("/model-releases-by-country")
def get_model_releases_by_country() -> list[dict]:
    """Return model releases count by country."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT 
                c.country_hq AS country,
                COUNT(m.id) AS model_count
            FROM companies c
            LEFT JOIN models m ON c.id = m.company_id
            WHERE c.country_hq IS NOT NULL
            GROUP BY c.country_hq
            ORDER BY model_count DESC
            """
        )
        return [dict(row) for row in cur.fetchall()]


@router.get("/model-releases-by-company")
def get_model_releases_by_company() -> list[dict]:
    """Return model releases count by company."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT 
                c.id AS company_id,
                c.display_name AS company_name,
                COUNT(m.id) AS model_count
            FROM companies c
            LEFT JOIN models m ON c.id = m.company_id
            GROUP BY c.id, c.display_name
            ORDER BY model_count DESC
            """
        )
        return [dict(row) for row in cur.fetchall()]


@router.get("/license-trends")
def get_license_trends() -> list[dict]:
    """Return license distribution trends."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT 
                COALESCE(l.slug, 'unknown') AS license_slug,
                COUNT(*) AS count
            FROM models m
            LEFT JOIN licenses l ON m.license_id = l.id
            GROUP BY l.slug
            ORDER BY count DESC
            """
        )
        return [dict(row) for row in cur.fetchall()]


@router.get("/country-comparison")
def get_country_comparison() -> list[dict]:
    """Return country comparison data."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT 
                c.country_hq,
                COUNT(DISTINCT c.id) AS company_count,
                COUNT(m.id) AS model_count
            FROM companies c
            LEFT JOIN models m ON c.id = m.company_id
            GROUP BY c.country_hq
            ORDER BY model_count DESC
            """
        )
        return [dict(row) for row in cur.fetchall()]


@router.get("/time-analysis")
def get_time_analysis() -> list[dict]:
    """Return time-based analysis data."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                DATE_TRUNC('month', m.created_at) AS month,
                COUNT(*) AS model_count
            FROM models m
            GROUP BY DATE_TRUNC('month', m.created_at)
            ORDER BY month
            """
        )
        return [dict(row) for row in cur.fetchall()]


@router.get("/historical-total")
def get_historical_total() -> list[dict]:
    """Return total model releases per month across all orgs."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                TO_CHAR(DATE_TRUNC('month', m.release_date), 'YYYY-MM') AS month,
                COUNT(*) AS model_count
            FROM models m
            WHERE m.release_date IS NOT NULL
            GROUP BY DATE_TRUNC('month', m.release_date)
            ORDER BY month
            """
        )
        return [dict(row) for row in cur.fetchall()]


@router.get("/historical-by-company")
def get_historical_by_company() -> list[dict]:
    """Return monthly model releases broken down by company."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                TO_CHAR(DATE_TRUNC('month', m.release_date), 'YYYY-MM') AS month,
                c.display_name AS company_name,
                COUNT(*) AS model_count
            FROM models m
            JOIN companies c ON m.company_id = c.id
            WHERE m.release_date IS NOT NULL
            GROUP BY DATE_TRUNC('month', m.release_date), c.display_name
            ORDER BY month, model_count DESC
            """
        )
        return [dict(row) for row in cur.fetchall()]


@router.get("/historical-by-country")
def get_historical_by_country() -> list[dict]:
    """Return monthly model releases broken down by company HQ country."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                TO_CHAR(DATE_TRUNC('month', m.release_date), 'YYYY-MM') AS month,
                c.country_hq AS country,
                COUNT(*) AS model_count
            FROM models m
            JOIN companies c ON m.company_id = c.id
            WHERE m.release_date IS NOT NULL
              AND c.country_hq IS NOT NULL
            GROUP BY DATE_TRUNC('month', m.release_date), c.country_hq
            ORDER BY month, model_count DESC
            """
        )
        return [dict(row) for row in cur.fetchall()]
