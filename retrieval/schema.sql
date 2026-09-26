CREATE TABLE IF NOT EXISTS documents (
    doc_id       UUID PRIMARY KEY,
    display_name TEXT NOT NULL UNIQUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

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
