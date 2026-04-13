@echo off
cd /d "%~dp0"
uvicorn api.app:app --reload --port 8000
