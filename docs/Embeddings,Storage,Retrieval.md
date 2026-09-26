# Embedding, Storage & Retrieval — Low-Level Design

**Project:** Niva Bupa Insurance Policy RAG
**Scope:** Three pipeline stages — (1) embedding child chunks, (2) storing chunks/parents/vectors, (3) query-time semantic retrieval. Chunking and table handling are in the Ingestion LLD; document lifecycle (dedup, versioning, upload) is in the File Upload LLD. Both are referenced only where they hand data into this stage.

**Status:** v1 — **semantic-only retrieval**. Keyword/hybrid search is deferred to v2 (see §5).

---

## 1. Embedding

### 1.1 Model
- **Provider:** Databricks Model Serving, endpoint `databricks-gte-large-en` (`gte-large-en-v1.5`), configured via `EMBEDDING_MODEL`, `DATABRICKS_HOST`, `DATABRICKS_TOKEN`.
- **Dimension:** 1024 (measured against the live endpoint). Fixed at S3 Vectors index creation; changing it means rebuilding the index.
- **Query vs. document encoding:** GTE-large-en needs **no** instruction prefix — the same text-in call is used for chunks and queries. If the model is ever swapped for one that needs prefixes (E5/BGE/Qwen3), `embed_query` / `embed_documents` are the single place to add them.

### 1.2 Token limits
- Model context is 8192 tokens; child chunks are capped at 500 tokens (`ingestion.config.MAX_TOKENS`). A config constant `EMBED_MAX_TOKENS` guards the call; a child over it is a bug upstream and fails loudly rather than being truncated silently.

### 1.3 Failure handling
- Transient errors (429/5xx/timeouts) retry with exponential backoff inside the client. Persistent failure raises, and the document follows the existing retry-from-start strategy capped by `retry_count`.

---

## 2. Storage

### 2.1 S3 Vectors
- One vector bucket, one index: dimension 1024, distance metric **cosine**, `float32`. Bucket/index names come from `S3_VECTORS_BUCKET` / `S3_VECTORS_INDEX`.

### 2.2 Vector metadata (filterable, reference only)

| Key | Type | Purpose |
|---|---|---|
| `doc_id` | String | `$in` filter target at query time |
| `parent_id` | String | Join key to Postgres `parents` |
| `chunk_key` | String | Row key; used to delete vectors (no delete-by-filter in S3 Vectors) |

No text is stored in vector metadata (2 KB filterable limit is far below parent size).

### 2.3 Postgres schema

**`documents`** — `doc_id` (UUID PK), `display_name` (unique, user-entered), `created_at`. Lifecycle columns (`content_hash`, `status`, `retry_count`, …) belong to the File Upload LLD and are added there.

**`parents`** — `parent_id` (PK), `doc_id` (FK), `parent_text`, plus `title`, `section`, `clauses`, `source_pages` for citations.

**`chunks`** — `chunk_key` (PK), `doc_id` (FK), `parent_id` (FK), `embedding_text` (exact text embedded), `kind`.

**ID namespacing.** The chunker numbers ids per file (`P0001`, `C0001` restart for every PDF). At load time they become `{doc_id}:P0001` / `{doc_id}:C0001`, so `parent_id` and `chunk_key` are globally unique. The chunker is unchanged.

### 2.4 Write path (per document, at load)
1. In one Postgres transaction: insert `documents`, `parents`, `chunks` (with `embedding_text` = child `text`).
2. Embed all children in batches (document encoding).
3. `PutVectors` in batches of ≤ 500 with `{doc_id, parent_id, chunk_key}` metadata.
4. If step 2 or 3 fails, the vectors written so far are deleted by `chunk_key` and the Postgres rows are rolled back, so a failed load leaves nothing behind and retry-from-start is safe.

Postgres is written first because `chunks` is the only reliable record of which keys exist for `DeleteVectors`.

---

## 3. Retrieval

```
Input: user_query (text), selected_policy (e.g. "ReAssure"), top_k

Step 1 — Pre-filter (Postgres)
    doc_ids = SELECT doc_id FROM documents WHERE display_name LIKE '<selected_policy>%'
    IF empty: return "no matching policy documents found", STOP
    (never call QueryVectors with an empty $in list)

Step 2 — Semantic search (S3 Vectors)
    query_vec = embed_query(user_query)
    hits = QueryVectors(index, query_vec, filter={"doc_id": {"$in": doc_ids}}, top_k)
    → ranked list of {chunk_key, parent_id, distance}

Step 3 — Dedup
    Walk hits in rank order, keep the first occurrence of each parent_id

Step 4 — Parent fetch (Postgres)
    SELECT parent_id, parent_text ... WHERE parent_id IN (<deduped ids>)

Step 5 — Assembly
    Return parents in rank order for injection into the LLM context
```

### 3.1 Why this shape
- **Prefix match runs in Postgres, not in S3 Vectors** — S3 Vectors has no prefix/regex operator (`$eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists, $and, $or`). Only the resolved exact `doc_id` list is passed via `$in`.
- **Dedup before parent fetch** — several children (rows of one lookup table, a clause and its sidebar paraphrase) resolve to one parent; deduping avoids injecting the same block twice.
- **`top_k` counts children, not parents** — after dedup fewer parents are returned. `top_k` is over-fetched relative to the parent count wanted.

---

## 4. Config constants

| Constant | Value | Status |
|---|---|---|
| Embedding dimension | 1024 | Locked |
| Distance metric | cosine | Locked |
| `top_k` (children) | 10 | Placeholder; tune with RAGAS |
| Embed batch size | 8 | Endpoint returns 429 at 16 inputs per call (pay-per-token QPS limit) |
| `PutVectors` batch size | 500 | S3 Vectors limit |

## 5. Deferred to v2 / open items

- **Keyword (BM25-style) leg and RRF fusion** — deferred. When revisited: `tsvector` column on `chunks` with GIN index, RRF k=60, and a `keyword_search_enabled` runtime flag to run RAGAS vector-only vs. hybrid. Note `to_tsquery` errors on raw natural-language input, and `plainto_tsquery` / `websearch_to_tsquery` AND every term, so long questions match nothing; the query must be built by OR-ing tokens.
- **`pg_search`/ParadeDB** — not available in-place on RDS/Aurora; only if `ts_rank` proves the bottleneck.
- **Enumeration queries** ("what are all the inclusions") — top-k over similar benefit children will omit items. Needs a fetch-all by `display_name` + `section` path (see Ingestion LLD, CIS note).
- **Parent token budget** — parents reach ~4k tokens; assembly may need a cap on total context.
- **`top_k`** — set from observed recall in RAGAS.
