"""
/api/licenses routes.

GET /api/licenses        — list all licenses with model counts
GET /api/licenses/{slug} — full detail: metadata, license text, companies using it
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.db import get_cursor

router = APIRouter(prefix="/api/licenses", tags=["licenses"])


@router.get("")
def list_licenses() -> list[dict]:
    """Return all licenses sorted by model count descending."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                l.id,
                l.slug,
                l.display_name,
                l.family,
                l.is_osi_approved,
                l.allows_commercial_use,
                l.allows_derivatives,
                l.requires_attribution,
                l.requires_share_alike,
                COUNT(m.id) AS model_count
            FROM licenses l
            LEFT JOIN models m ON m.license_id = l.id
            GROUP BY l.id
            ORDER BY model_count DESC, l.slug
            """
        )
        return [dict(row) for row in cur.fetchall()]


@router.get("/{slug}")
def get_license(slug: str) -> dict:
    """Return full detail for a single license, including text and companies using it."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                id, slug, display_name, family,
                is_osi_approved,
                allows_commercial_use, allows_derivatives,
                requires_attribution, requires_share_alike,
                notes, license_text, source_url, created_at
            FROM licenses
            WHERE slug = %s
            """,
            (slug,),
        )
        lic = cur.fetchone()
        if lic is None:
            raise HTTPException(status_code=404, detail="License not found")

        # Companies using this license with their models
        cur.execute(
            """
            SELECT
                c.id           AS company_id,
                c.display_name AS company_name,
                COUNT(m.id)    AS model_count,
                json_agg(
                    json_build_object(
                        'model_id',    m.model_id,
                        'display_name', COALESCE(m.display_name, m.model_id),
                        'url',         m.url,
                        'release_date', m.release_date
                    )
                    ORDER BY m.release_date DESC NULLS LAST
                ) AS models
            FROM models m
            JOIN companies c ON c.id = m.company_id
            WHERE m.license_id = %s
            GROUP BY c.id, c.display_name
            ORDER BY model_count DESC
            """,
            (lic["id"],),
        )
        companies = [dict(row) for row in cur.fetchall()]

    result = dict(lic)
    result["companies"] = companies
    return result
