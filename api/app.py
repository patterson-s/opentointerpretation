"""
opentointerpretation — FastAPI application entry point.

Run:
    uvicorn api.app:app --reload --port 8080   (local dev)
    uvicorn api.app:app --host 0.0.0.0 --port $PORT  (production / Render)
"""

from __future__ import annotations

import base64
import os
import secrets
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from api.routers import companies, analysis, licenses, models, research, status


# ---------------------------------------------------------------------------
# HTTP Basic Auth middleware — protects every route (API + static files).
# Credentials are read from env vars so they can be rotated without a
# code change.  Defaults match the project's initial public credentials.
# ---------------------------------------------------------------------------

class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        expected_user = os.environ.get("BASIC_AUTH_USERNAME", "spatt")
        expected_pass = os.environ.get("BASIC_AUTH_PASSWORD", "spatt")
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Basic "):
            try:
                decoded = base64.b64decode(auth[6:]).decode("utf-8", errors="replace")
                user, _, pw = decoded.partition(":")
                ok_user = secrets.compare_digest(user, expected_user)
                ok_pass = secrets.compare_digest(pw, expected_pass)
                if ok_user and ok_pass:
                    return await call_next(request)
            except Exception:
                pass
        return Response(
            "Unauthorized",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Open to Interpretation"'},
        )


app = FastAPI(
    title="Open to Interpretation API",
    description="Explore AI lab licensing data",
    version="0.1.0",
)

# Basic auth — applied first so every request is gated
app.add_middleware(BasicAuthMiddleware)

# CORS — allow all origins (site is password-protected; Render URL is unknown at build time)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# API routers
app.include_router(companies.router)
app.include_router(analysis.router)
app.include_router(licenses.router)
app.include_router(models.router)
app.include_router(research.router)
app.include_router(status.router)

# Serve the web frontend — mount AFTER API routes so /api/* takes priority
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"
app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
