CREATE TABLE IF NOT EXISTS documents (
    doc_id          UUID PRIMARY KEY,
    display_name    TEXT NOT NULL,
    content_hash    TEXT,
    s3_key          TEXT,
    status          TEXT NOT NULL DEFAULT 'indexed',
    retry_count     INT NOT NULL DEFAULT 0,
    error_message   TEXT,
    replaces_doc_id UUID,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Upgrade a table created by an earlier version of this file.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS s3_key TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'indexed';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS retry_count INT NOT NULL DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS error_message TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS replaces_doc_id UUID;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_display_name_key;

-- Names and content are unique among live documents only. A document that replaces
-- another carries replaces_doc_id until the old one is deleted, so it may share its name.
CREATE UNIQUE INDEX IF NOT EXISTS documents_name_live_idx
    ON documents (display_name) WHERE status <> 'deleted' AND replaces_doc_id IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS documents_hash_live_idx
    ON documents (content_hash) WHERE status <> 'deleted';

CREATE TABLE IF NOT EXISTS parents (
    parent_id    TEXT PRIMARY KEY,
    doc_id       UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    parent_text  TEXT NOT NULL,
    title        TEXT,
    section      TEXT,
    clauses      JSONB NOT NULL DEFAULT '[]',
    source_pages JSONB NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_key      TEXT PRIMARY KEY,
    doc_id         UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    parent_id      TEXT NOT NULL REFERENCES parents(parent_id) ON DELETE CASCADE,
    embedding_text TEXT NOT NULL,
    kind           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS parents_doc_id_idx ON parents (doc_id);
