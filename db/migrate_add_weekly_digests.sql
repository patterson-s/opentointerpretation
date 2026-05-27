-- Migration: add weekly_digests table for Home tab digest reports
-- Safe to re-run: uses IF NOT EXISTS / CREATE TABLE IF NOT EXISTS

CREATE TABLE IF NOT EXISTS weekly_digests (
    id           SERIAL PRIMARY KEY,
    week_start   DATE NOT NULL,
    week_end     DATE NOT NULL,
    new_models   INT  NOT NULL DEFAULT 0,
    by_company   JSONB NULL,   -- [{name, hf_handle, count}, ...] sorted by count desc
    by_license   JSONB NULL,   -- [{slug, count}, ...] sorted by count desc
    by_modality  JSONB NULL,   -- [{modality, count}, ...] sorted by count desc
    narrative    TEXT NULL,    -- Claude-generated 3-sentence digest
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (week_start)
);

CREATE INDEX IF NOT EXISTS idx_weekly_digests_week_start
    ON weekly_digests (week_start DESC);
