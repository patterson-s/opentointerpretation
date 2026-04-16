-- Migration: add collection_runs table for incremental HF data collection
-- Run once: psql -U postgres -d opentointerpretation -f db/migrate_add_collection_runs.sql

CREATE TABLE IF NOT EXISTS collection_runs (
    id                SERIAL PRIMARY KEY,
    started_at        TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    completed_at      TIMESTAMPTZ   NULL,
    status            VARCHAR(20)   NOT NULL DEFAULT 'running',
    total_models_seen INT           NULL,
    new_models_added  INT           NULL,
    error_message     TEXT          NULL
);

CREATE INDEX IF NOT EXISTS idx_collection_runs_started_at
    ON collection_runs (started_at DESC);
