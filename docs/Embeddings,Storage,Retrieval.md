# Embedding, Storage & Retrieval — Low-Level Design

**Project:** Niva Bupa Insurance Policy RAG
**Scope:** This LLD covers three pipeline stages only — (1) generating embeddings for child chunks, (2) storing chunks/parents/vectors, (3) query-time hybrid retrieval. Chunking strategy, table-handling rules, and document lifecycle (dedup, versioning) are covered in the separate Ingestion LLD and are referenced here only where they hand off data into this stage.

**Status:** Draft — reflects decisions locked as of this conversation. Open items are called out explicitly in Section 5 rather than presented as resolved.

---

## 1. Embedding

### 1.1 Model
- **Provider:** Databricks Model Serving (Foundation Model API) embedding endpoint.
- **Dimension:** Fixed at whichever Databricks-hosted model is selected. Must be pinned **before** the S3 Vectors index is created — vector index dimension cannot be changed post-creation without rebuilding the index.
- **Query vs. document encoding:** Confirm whether the selected model exposes separate query/document input modes via API parameter, or requires instruction-prefix convention in the text itself (e.g. `"query: "` / `"passage: "` — common for E5/BGE/GTE-family models commonly hosted on Databricks). Whichever mechanism applies must be used consistently: child-chunk text embedded at ingestion uses the **document** encoding; user query text embedded at retrieval time uses the **query** encoding.

### 1.2 Token limits
- Apply the same pre-flight token-count guard pattern already used for the Cohere v3 ceiling risk: check token count of the synthesized child-chunk text before calling the embedding endpoint, and apply the sub-child splitting rule if the model's context ceiling would be exceeded. Exact ceiling value depends on the specific Databricks model chosen — confirm and hardcode as a config constant, not inferred at runtime.

### 1.3 Failure handling
- Embedding call failures during ingestion follow the existing retry-from-start strategy capped by `retry_count` (per document lifecycle design) — no new retry mechanism introduced here.

---

## 2. Storage

### 2.1 S3 Vector bucket & index
- One vector bucket (region-scoped, per existing AWS-native architecture).
- One vector index (dimension fixed per §1.1, distance metric: **cosine**).
- Encryption: SSE-S3 by default, SSE-KMS if a customer-managed key requirement applies.

### 2.2 Vector metadata schema (filterable)
Stored alongside each child chunk's embedding in the S3 Vectors index:

| Key | Type | Purpose |
|---|---|---|
| `doc_id` | String | Join key back to Postgres `documents`/`chunks` tables; used as the `$in` filter target at query time |
| `parent_id` | String | Join key to Postgres `parents` table for context assembly |
| `chunk_key` | String | Unique row key; used by the `DeleteVectors` workaround (S3 Vectors has no delete-by-filter) |

No document/parent **content** is stored in vector metadata — total metadata per vector is capped at 40 KB (filterable + non-filterable) and filterable metadata at 2 KB, which parent-level text (full clause blocks, multi-row tables) will routinely exceed. Metadata here is reference-only.

### 2.3 Postgres schema

**`documents`** (existing, extended if needed)
- `doc_id` (UUID, PK)
- `file_name` (unique, DB-enforced, user-entered per upload)
- other lifecycle columns per Document Lifecycle LLD (content hash, `is_update`, `replaces_doc_id`, etc.)

**`chunks`** (existing, extended)
- `chunk_key` (PK)
- `doc_id` (FK → documents)
- `parent_id` (FK → parents)
- `embedding_text` (the synthesized text that was embedded)
- `search_vector` (`tsvector`, generated from `embedding_text`) — new column for BM25

**`parents`** (new)
- `parent_id` (PK)
- `doc_id` (FK → documents)
- `parent_text` (full clause block or full table markdown — the actual context injected into the LLM)

### 2.4 Write path (per child chunk, at ingestion)
Single logical write, same transaction boundary as existing chunk insert:
1. Insert/verify `parents` row for the chunk's parent block (if not already written for this parent_id).
2. Insert `chunks` row: `chunk_key`, `doc_id`, `parent_id`, `embedding_text`, and `search_vector = to_tsvector('simple', embedding_text)`.
3. Call Databricks embedding endpoint (document mode) on `embedding_text`.
4. `PutVectors` into S3 Vectors index with the embedding + `{doc_id, parent_id, chunk_key}` metadata.

`'simple'` tsvector config chosen over `'english'` to avoid stemming/stopword removal distorting exact clause numbers and defined insurance terms (e.g. "Sum Insured"). To be validated against real query patterns before final lock (see §5).

Index: `CREATE INDEX chunks_search_idx ON chunks USING GIN (search_vector);`

---

## 3. Retrieval

### 3.1 Step-by-step flow

```
Input: user_query (text), selected_policy (e.g. "ReAssure")

Step 1 — Pre-filter (Postgres)
    doc_ids = SELECT doc_id FROM documents
              WHERE file_name LIKE '<selected_policy>%'

    IF doc_ids is empty:
        → return "no matching policy documents found", STOP
        (do not call QueryVectors with an empty $in list — invalid input)

Step 2 — Semantic leg (S3 Vectors)
    query_vec = embed(user_query, mode=query)
    results_A = QueryVectors(
        index,
        vector = query_vec,
        filter = { "doc_id": { "$in": doc_ids } },
        top_k = N
    )
    → ranked list of {chunk_key, parent_id, distance}

Step 3 — Keyword leg (Postgres tsvector) — SKIPPED if keyword_search_enabled = false
    IF keyword_search_enabled:
        tsquery = to_tsquery('simple', user_query)
        results_B = SELECT chunk_key, parent_id, ts_rank(search_vector, tsquery) AS rank
                    FROM chunks
                    WHERE doc_id = ANY(doc_ids)
                      AND search_vector @@ tsquery
                    ORDER BY rank DESC
                    LIMIT N
        → ranked list of {chunk_key, parent_id, rank}
    ELSE:
        results_B = [] (query not executed — no Postgres FTS call made)

Step 4 — Fusion (Reciprocal Rank Fusion, k=60) — SKIPPED if keyword_search_enabled = false
    IF keyword_search_enabled:
        For each chunk_key appearing in results_A and/or results_B:
            rrf_score(chunk_key) = Σ 1 / (60 + rank_in_list)
                                    (summed over whichever list/lists it appears in)
        → single ranked list of chunk_keys, sorted by rrf_score desc
    ELSE:
        final_ranked = results_A, in existing distance-ranked order (no fusion pass)

Step 5 — Dedup
    Walk the fused ranked list, collect distinct parent_ids in rank order
    (first occurrence wins; drop repeats of a parent_id already collected)

Step 6 — Parent fetch (Postgres)
    parent_texts = SELECT parent_id, parent_text FROM parents
                   WHERE parent_id IN (<deduped parent_ids>)

Step 7 — Assembly
    Inject parent_texts (in fused-rank order) into LLM context
```

### 3.2 Config flag: `keyword_search_enabled`

A single boolean controls whether the keyword leg runs at all.

- **`true`**: full hybrid flow as in §3.1 — both legs run, RRF fusion applied.
- **`false`**: Step 3's Postgres FTS query is not executed at all (not run-and-discarded), Step 4's fusion pass is skipped, and `results_A` (semantic leg, already ranked by distance) is passed straight through as the final ranked list.
- **Scope**: passed as a runtime parameter to the retrieval function/API call (sourced from a static app-config default, but overridable per call). This is what allows running the same code path against the same corpus with the flag flipped both ways — the intended mechanism for the RAGAS vector-only-vs-hybrid comparison in §5, without maintaining two separate retrieval implementations or redeploying between runs.
- **Ingestion is unaffected by this flag** — `search_vector` is always populated for every chunk at write time (§2.4), regardless of the flag's current value. This keeps the column ready the moment the flag is flipped on for an eval run, rather than requiring re-ingestion of the corpus to backfill it.

### 3.3 Why this shape
- **Pre-filter uses `LIKE` on `file_name`, not on S3 Vectors metadata** — S3 Vectors' supported filter operators are `$eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists, $and, $or`; there is no prefix/regex operator. Prefix resolution therefore happens in Postgres, and only the resolved exact `doc_id` list is passed to S3 Vectors via `$in`.
- **Both retrieval legs are constrained to the same `doc_id` list** before fusion — prevents the keyword leg from surfacing exact-term matches from a policy the semantic leg was excluded from, and vice versa.
- **RRF over raw score blending** — cosine distance and `ts_rank` are non-comparable scales; RRF uses rank position only, avoiding score-normalization tuning.
- **Dedup happens after fusion, before parent fetch** — several matching child rows (e.g. multiple rows of the same lookup table) commonly resolve to one parent; deduping first avoids injecting the same parent block multiple times and blowing up context-window/token cost.

---

## 4. Config constants to finalize

| Constant | Value here | Status |
|---|---|---|
| Embedding dimension | model-dependent | Pin before index creation |
| Distance metric | cosine | Locked |
| `top_k` per leg (N) | not yet set | Set based on RAGAS results |
| RRF `k` | 60 (standard default) | Can tune if fusion quality needs it |
| tsvector config | `'simple'` | Validate vs `'english'` on real queries |
| `keyword_search_enabled` | default TBD (`false` until eval decides) | Runtime-overridable flag; drives the RAGAS vector-only-vs-hybrid comparison in §5 |

## 5. Open items (explicitly not decided here)

- **`top_k` per leg** — needs setting based on observed recall in RAGAS evaluation, not hardcoded arbitrarily.
- **`'simple'` vs `'english'` tsvector config** — flagged as the better default for clause numbers/defined terms, but not yet validated against actual query logs.
- **Query expansion / synonym handling for the BM25 leg** — not designed here. If exact-term misses show up in evaluation (e.g. abbreviations, alternate phrasings), this would need a separate design pass — not assumed as part of this LLD.
- **Whether hybrid (vs. semantic-only) is justified at all** — the `keyword_search_enabled` flag (§3.2) is the mechanism for resolving this: run the same retrieval code through RAGAS with the flag set `true` and `false` on the real corpus, and let the observed comparison decide the production default, rather than committing to hybrid purely on the theoretical lexical-search argument.
- **`pg_search`/ParadeDB (true BM25 scoring) was evaluated and explicitly deferred** — it isn't supported as an in-place extension on AWS RDS/Aurora, requiring either a self-hosted logical-replication sidecar or migrating off managed Postgres entirely. Given `tsvector`/`ts_rank` is a reasonable approximation of lexical ranking and avoids new infrastructure, this LLD stays with `tsvector` for v1; revisit only if RAGAS results show the ranking approximation (not hybrid-vs-semantic itself) is the bottleneck.