-- Migration: add raw model card storage to models table
-- Safe to re-run: uses ADD COLUMN IF NOT EXISTS
-- Apply: psql -U postgres -d opentointerpretation -f db/migrate_add_model_card.sql

BEGIN;

ALTER TABLE models
    ADD COLUMN IF NOT EXISTS model_card_raw        TEXT NULL,
    ADD COLUMN IF NOT EXISTS model_card_fetched_at TIMESTAMPTZ NULL;

COMMIT;
