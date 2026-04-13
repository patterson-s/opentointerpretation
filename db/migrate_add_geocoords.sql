-- Add geocoordinate columns to research_findings for map visualization.
-- Run: psql -U postgres -d opentointerpretation -f db/migrate_add_geocoords.sql
-- Idempotent: uses ADD COLUMN IF NOT EXISTS.

ALTER TABLE research_findings
  ADD COLUMN IF NOT EXISTS latitude  FLOAT NULL,
  ADD COLUMN IF NOT EXISTS longitude FLOAT NULL;
