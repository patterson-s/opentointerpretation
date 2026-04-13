-- Research provenance schema: sources, chunks (with embeddings), findings
-- Run: psql -U postgres -d opentointerpretation -f db/migrate_research_tables.sql
-- Idempotent: all CREATE statements use IF NOT EXISTS.

-- Raw search results and fetched web pages, affiliated to a company
CREATE TABLE IF NOT EXISTS research_sources (
    id              SERIAL PRIMARY KEY,
    company_id      INT NULL REFERENCES companies(id) ON DELETE SET NULL,
    url             VARCHAR(2000) NOT NULL,
    title           VARCHAR(1000) NULL,
    snippet         TEXT NULL,           -- Serper result snippet
    raw_content     TEXT NULL,           -- full fetched + extracted text
    search_query    VARCHAR(500) NULL,   -- query that surfaced this source
    source_type     VARCHAR(50)  NULL,   -- 'serper_result' | 'web_fetch'
    fetched_at      TIMESTAMPTZ  DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_research_sources_url
    ON research_sources(url);
CREATE INDEX IF NOT EXISTS idx_research_sources_company
    ON research_sources(company_id);

-- Text chunks derived from sources, with Cohere embed-v4.0 1024-dim embeddings
-- Stored as FLOAT[] (native PostgreSQL array) since pgvector is not required.
-- Similarity search is done in Python (numpy cosine).
CREATE TABLE IF NOT EXISTS research_chunks (
    id              SERIAL PRIMARY KEY,
    source_id       INT NOT NULL REFERENCES research_sources(id) ON DELETE CASCADE,
    chunk_index     INT NOT NULL,
    chunk_text      TEXT NOT NULL,
    embedding       FLOAT[] NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_research_chunks_source
    ON research_chunks(source_id);

-- Extracted location findings (headquarters + affiliate offices) per company
CREATE TABLE IF NOT EXISTS research_findings (
    id              SERIAL PRIMARY KEY,
    company_id      INT NULL REFERENCES companies(id) ON DELETE SET NULL,
    finding_type    VARCHAR(50) NOT NULL,   -- 'headquarters' | 'affiliate'
    city            VARCHAR(255) NULL,
    country         VARCHAR(255) NULL,
    confidence      VARCHAR(20) NULL,       -- 'confirmed' | 'unconfirmed'
    source_count    INT DEFAULT 0,          -- number of sources substantiating this
    notes           TEXT NULL,
    extracted_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (company_id, finding_type, city, country)
);
CREATE INDEX IF NOT EXISTS idx_research_findings_company
    ON research_findings(company_id);

-- Junction table: which sources support which findings
CREATE TABLE IF NOT EXISTS research_finding_sources (
    finding_id  INT NOT NULL REFERENCES research_findings(id) ON DELETE CASCADE,
    source_id   INT NOT NULL REFERENCES research_sources(id) ON DELETE CASCADE,
    PRIMARY KEY (finding_id, source_id)
);
