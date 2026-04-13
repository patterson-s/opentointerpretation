"""
opentointerpretation — FastAPI application entry point.

Run:
    uvicorn api.app:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routers import companies, analysis, licenses, models, research

app = FastAPI(
    title="Open to Interpretation API",
    description="Explore AI lab licensing data",
    version="0.1.0",
)

# CORS — allow localhost dev ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# API routers
app.include_router(companies.router)
app.include_router(analysis.router)
app.include_router(licenses.router)
app.include_router(models.router)
app.include_router(research.router)

# Serve the web frontend — mount AFTER API routes so /api/* takes priority
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"
app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
