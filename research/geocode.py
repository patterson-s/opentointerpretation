"""
Geocode research_findings city/country pairs using Nominatim (OpenStreetMap).

Writes latitude and longitude back to the research_findings table.
Skips rows that already have coordinates (idempotent).
Skips rows where city starts with "Various" (Baidu regional placeholders).

Usage:
    python research/geocode.py             # geocode all missing rows
    python research/geocode.py --force     # re-geocode all rows
    python research/geocode.py --dry-run   # print without writing

Rate limit: 1 second between Nominatim requests (OSM usage policy).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Force UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_ENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(_ENV_PATH, override=True)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_HEADERS = {
    "User-Agent": "opentointerpretation-research/1.0 (academic research project)"
}

# Special-case city name overrides for geocoding queries
CITY_OVERRIDES: dict[str, str] = {
    "Silicon Valley": "San Jose, California",
    "USA":           "",   # country-only, skip
    "Europe":        "",
    "Latin America": "",
    "Middle East":   "",
    "Africa":        "",
}


def _connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ.get("PGPORT", 5432)),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ.get("PGPASSWORD", ""),
    )


def geocode(city: str, country: str) -> tuple[Optional[float], Optional[float]]:
    """
    Query Nominatim for (city, country) and return (lat, lng) or (None, None).
    """
    # Apply overrides
    city_q = CITY_OVERRIDES.get(city, city)
    if not city_q:
        return None, None

    query = f"{city_q}, {country}"
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": query, "format": "json", "limit": 1},
            headers=NOMINATIM_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
        # Try without city if initial query failed
        resp2 = requests.get(
            NOMINATIM_URL,
            params={"q": country, "format": "json", "limit": 1},
            headers=NOMINATIM_HEADERS,
            timeout=10,
        )
        data2 = resp2.json()
        if data2:
            return float(data2[0]["lat"]), float(data2[0]["lon"])
    except Exception as e:
        print(f"  Nominatim error for '{query}': {e}", file=sys.stderr)
    return None, None


def run(force: bool = False, dry_run: bool = False) -> None:
    conn = _connect()
    failures: list[str] = []

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if force:
                cur.execute("SELECT id, city, country FROM research_findings ORDER BY id")
            else:
                cur.execute(
                    "SELECT id, city, country FROM research_findings "
                    "WHERE latitude IS NULL ORDER BY id"
                )
            rows = cur.fetchall()

        print(f"Geocoding {len(rows)} finding(s)...")

        for row in rows:
            fid   = row["id"]
            city  = (row["city"] or "").strip()
            country = (row["country"] or "").strip()

            # Skip vague/regional entries
            if city.startswith("Various") or not city or not country:
                print(f"  [{fid}] SKIP  {city}, {country}")
                continue

            lat, lng = geocode(city, country)
            time.sleep(1.0)  # OSM rate-limit: 1 req/sec

            if lat is None:
                print(f"  [{fid}] FAIL  {city}, {country}")
                failures.append(f"{city}, {country}")
                continue

            print(f"  [{fid}] OK    {city}, {country} -> ({lat:.4f}, {lng:.4f})")

            if not dry_run:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE research_findings SET latitude=%s, longitude=%s WHERE id=%s",
                        (lat, lng, fid),
                    )
                conn.commit()

        print(f"\nDone. Failures ({len(failures)}):")
        for f in failures:
            print(f"  {f}")

    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Geocode research_findings city/country pairs.")
    parser.add_argument("--force",   action="store_true", help="Re-geocode all rows")
    parser.add_argument("--dry-run", action="store_true", help="Print without writing to DB")
    args = parser.parse_args()
    run(force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
