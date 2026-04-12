-- opentointerpretation: PostgreSQL schema
-- Levels: companies -> models <- licenses

-- ── companies ──────────────────────────────────────────────────────────────
-- One row per organization. hf_handle is NULL for closed-source-only orgs.
CREATE TABLE IF NOT EXISTS companies (
    id           SERIAL PRIMARY KEY,
    display_name VARCHAR(510)  NOT NULL,
    hf_handle    VARCHAR(100)  NULL,
    country_hq   VARCHAR(255)  NULL,
    created_at   TIMESTAMPTZ   DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_companies_display_name
    ON companies (display_name);

-- ── licenses ───────────────────────────────────────────────────────────────
-- One row per license slug. Metadata columns (family, is_osi_approved, notes)
-- can be enriched later without schema changes.
CREATE TABLE IF NOT EXISTS licenses (
    id              SERIAL PRIMARY KEY,
    slug            VARCHAR(510)  NOT NULL,
    display_name    VARCHAR(510)  NULL,
    family          VARCHAR(100)  NULL,   -- 'open-source' | 'bespoke' | 'proprietary' | 'unknown'
    is_osi_approved BOOLEAN       NULL,
    notes           TEXT          NULL,
    created_at      TIMESTAMPTZ   DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_licenses_slug
    ON licenses (slug);

-- ── models ─────────────────────────────────────────────────────────────────
-- One row per model release. Covers both HuggingFace open models and
-- closed-source version history (OpenAI, Anthropic, Google).
-- metadata JSONB column provides an extension point for future fields.
CREATE TABLE IF NOT EXISTS models (
    id             SERIAL PRIMARY KEY,
    company_id     INT           NULL REFERENCES companies(id) ON DELETE SET NULL,
    license_id     INT           NULL REFERENCES licenses(id)  ON DELETE SET NULL,
    model_id       VARCHAR(1000) NOT NULL,  -- canonical ID: "handle/name" or "gpt-4"
    display_name   VARCHAR(1000) NULL,      -- human-readable variant (e.g. "GPT-4 Turbo")
    url            VARCHAR(1000) NULL,
    data_source    VARCHAR(50)   NOT NULL,  -- 'huggingface' | 'openai' | 'anthropic' | 'google'
    generation     VARCHAR(510)  NULL,      -- e.g. "GPT-4", "Claude 3"
    api_id         VARCHAR(510)  NULL,      -- Anthropic API identifier (claude-3-haiku-20240307)
    release_date   DATE          NULL,
    context_tokens BIGINT        NULL,
    likes          INT           NULL,
    downloads      BIGINT        NULL,
    notes          TEXT          NULL,
    metadata       JSONB         NULL,      -- extensible: new fields go here first
    created_at     TIMESTAMPTZ   DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_models_model_id
    ON models (model_id);
CREATE INDEX IF NOT EXISTS idx_models_company_id
    ON models (company_id);
CREATE INDEX IF NOT EXISTS idx_models_license_id
    ON models (license_id);
CREATE INDEX IF NOT EXISTS idx_models_data_source
    ON models (data_source);
