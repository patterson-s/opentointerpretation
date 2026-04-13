"""
/api/models routes.

GET /api/models          — paginated, filtered list
GET /api/models/filters  — distinct filter option values (countries, modalities)
GET /api/models/{model_id:path} — full detail for one model
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from api.db import get_cursor

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/filters")
def get_model_filters() -> dict:
    """Return distinct filter values for the country and modality dropdowns."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT country_hq
            FROM companies
            WHERE country_hq IS NOT NULL
            ORDER BY country_hq
            """
        )
        countries = [row["country_hq"] for row in cur.fetchall()]

        cur.execute(
            """
            SELECT DISTINCT metadata->>'modality' AS modality
            FROM models
            WHERE metadata->>'modality' IS NOT NULL
            ORDER BY modality
            """
        )
        modalities = [row["modality"] for row in cur.fetchall()]

    return {"countries": countries, "modalities": modalities}


@router.get("")
def list_models(
    company_id: Optional[int] = None,
    license_slug: Optional[str] = None,
    country_hq: Optional[str] = None,
    data_source: Optional[str] = None,
    modality: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Return a paginated, filtered list of models."""
    conditions: list[str] = []
    params: list = []

    if company_id is not None:
        conditions.append("m.company_id = %s")
        params.append(company_id)
    if license_slug is not None:
        conditions.append("l.slug = %s")
        params.append(license_slug)
    if country_hq is not None:
        conditions.append("co.country_hq = %s")
        params.append(country_hq)
    if data_source is not None:
        conditions.append("m.data_source = %s")
        params.append(data_source)
    if modality is not None:
        conditions.append("m.metadata->>'modality' = %s")
        params.append(modality)

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM models m
            LEFT JOIN companies co ON m.company_id = co.id
            LEFT JOIN licenses  l  ON m.license_id  = l.id
            {where_clause}
            """,
            params,
        )
        total = cur.fetchone()["count"]

        cur.execute(
            f"""
            SELECT
                m.id,
                m.model_id,
                COALESCE(m.display_name, m.model_id)          AS display_name,
                co.display_name                                AS company_name,
                l.slug                                         AS license_slug,
                m.data_source,
                m.metadata->>'modality'                        AS modality,
                CAST(m.metadata->>'num_parameters' AS DOUBLE PRECISION) AS num_parameters,
                m.release_date,
                m.downloads,
                m.likes
            FROM models m
            LEFT JOIN companies co ON m.company_id = co.id
            LEFT JOIN licenses  l  ON m.license_id  = l.id
            {where_clause}
            ORDER BY m.release_date DESC NULLS LAST, m.id DESC
            LIMIT %s OFFSET %s
            """,
            params + [limit, offset],
        )
        rows = [dict(row) for row in cur.fetchall()]

    return {"total": total, "models": rows}


@router.get("/{model_id:path}")
def get_model(model_id: str) -> dict:
    """Return full detail for a single model by its canonical model_id."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                m.id,
                m.model_id,
                m.display_name,
                m.url,
                m.data_source,
                m.generation,
                m.release_date,
                m.context_tokens,
                m.likes,
                m.downloads,
                m.notes,
                m.metadata,
                co.display_name  AS company_name,
                co.id            AS company_id,
                co.country_hq,
                co.hf_handle,
                l.slug           AS license_slug,
                l.display_name   AS license_display_name,
                l.family         AS license_family,
                l.is_osi_approved
            FROM models m
            LEFT JOIN companies co ON m.company_id = co.id
            LEFT JOIN licenses  l  ON m.license_id  = l.id
            WHERE m.model_id = %s
            """,
            (model_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Model not found")

    return dict(row)
