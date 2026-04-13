-- Migration: add license text storage and analysis columns to licenses table
-- Safe to re-run: uses IF NOT EXISTS
-- Apply: psql -U postgres -d opentointerpretation -f db/migrate_add_license_text.sql

BEGIN;

ALTER TABLE licenses
    ADD COLUMN IF NOT EXISTS license_text           TEXT NULL,
    ADD COLUMN IF NOT EXISTS source_url             VARCHAR(1000) NULL,
    ADD COLUMN IF NOT EXISTS allows_commercial_use  BOOLEAN NULL,
    ADD COLUMN IF NOT EXISTS allows_derivatives     BOOLEAN NULL,
    ADD COLUMN IF NOT EXISTS requires_attribution   BOOLEAN NULL,
    ADD COLUMN IF NOT EXISTS requires_share_alike   BOOLEAN NULL;

COMMIT;
