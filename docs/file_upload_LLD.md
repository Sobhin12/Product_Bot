# File Upload — Low Level Design

## Overview
User-facing file upload for a production-grade RAG system. FastAPI owns validation, dedup, S3 upload, and in-process async ingestion. Persistence is PostgreSQL; vector storage is Amazon S3 Vectors.

**Document pool is common/shared** — there is no per-user scoping or ownership. All authenticated users upload into, list, and manage the same single document set. `user_id` is used only for authentication and per-user rate limiting on the endpoint itself; it is **not** stored on the `documents` table and does not affect dedup, filename uniqueness, or listing.

Ingestion runs **without Celery or a message broker (no RabbitMQ/Redis-as-broker)** — it is scheduled as an in-process async task with an explicit concurrency cap (see "Ingestion Execution Model" below). This is a deliberate v1 simplification; a durable task queue is deferred to v2 (see Open Items).

---

## Document ID Generation & S3 Key Convention

**`document_id`** — UUIDv4, generated in the `/upload` handler before any DB write, so the same value can be used for the DB row and the S3 key without a round-trip dependency.

```python
import uuid
document_id = str(uuid.uuid4())
```

**S3 key convention:**
```
s3://{bucket}/documents/{document_id}/{display_name}
```
`display_name` is the user-entered, globally-unique name (see `POST /upload`), used here instead of the raw original filename since it's already guaranteed unique and is the identifier users actually recognize. On replace, the new document gets a fresh `document_id` and a fresh S3 path; the old document's S3 object is removed once the new one is confirmed indexed.

---

## Chunk Key Tracking (`chunks` table)

**Why this exists:** Amazon S3 Vectors' `DeleteVectors` API only accepts an explicit list of vector `keys` — it does **not** support deleting by metadata filter the way Qdrant did. `QueryVectors` (the API that does support metadata filters) is a nearest-neighbor search bounded by a top-K result limit, not a guaranteed-exhaustive scan, so it can't be relied on at delete-time to reliably enumerate every vector belonging to a `doc_id` — a document with more chunks than the top-K limit could have some chunks silently missed.

To make "delete all chunks for this `doc_id`" a reliable operation, chunk keys are tracked explicitly in PostgreSQL as they're written, not discovered later via a filter query.

```sql
CREATE TABLE chunks (
    chunk_key   TEXT PRIMARY KEY,   -- the vector's key in Amazon S3 Vectors
    doc_id      UUID NOT NULL REFERENCES documents(document_id),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_chunks_doc_id ON chunks (doc_id);
```

- **Written during `embed`:** each chunk's key is inserted into `chunks` at the same time (or immediately after) its vector is written via `PutVectors`
- **Read during delete:** `SELECT chunk_key FROM chunks WHERE doc_id = ?` returns the complete, authoritative list — no reliance on S3 Vectors' own query/filter behavior for enumeration
- **Batched delete:** the returned keys are deleted via `DeleteVectors` in batches of up to 500 (its documented per-call maximum)
- **Cleanup:** the corresponding `chunks` rows are removed (or cascade-deleted via the FK) once the vector deletion succeeds

---

## Endpoints

### `POST /upload`
Authenticated (any logged-in user may upload). Rate-limited per-user (same token-bucket mechanism as the query endpoint, applied here separately since `/upload` is now a real user-facing endpoint, not a trusted backend-only call).

**Request fields:**
- `file` — the document bytes
- `display_name` — user-entered name for the document; **must be unique** across the entire document pool (not per-user — there is no per-user scoping)
- `is_update` (bool) — whether this replaces an existing document
- `replaces_doc_id` (required if `is_update=true`) — the existing `document_id` being replaced

**Flow:**
1. Validate request — file size limit, content-type/extension allowlist (reject before touching disk)
2. Check `display_name` for uniqueness against PostgreSQL, filtered to non-`deleted` documents
   - If a conflict exists and `is_update=false` → reject with `409 Conflict` ("a document with this name already exists — choose another name or replace the existing one")
   - If `is_update=true`, the name conflict is expected to be with `replaces_doc_id` itself (same document, new content) — no rejection in that case
3. Save file to temp local disk path
4. Compute file hash (SHA-256) — before any S3 write
5. Check hash against PostgreSQL, filtered to non-`deleted` documents (global — no per-user scoping)
   - **If duplicate found:**
     - If `is_update=false` → return the existing `document_id`, `status=duplicate`, skip entirely — no S3 write, no DB row, no ingestion scheduled
     - If `is_update=true` and the hash matches `replaces_doc_id` itself (content unchanged) → return `status=no_changes`, skip entirely
   - **If no duplicate** → proceed
6. Generate `document_id` (UUIDv4)
7. Create DB row → `document_id`, `display_name`, `content_hash`, `status=pending`, `retry_count=0` (always a **new** `document_id`, even on replace — no version lineage tracked)
8. Schedule ingestion as an async task, subject to the concurrency limiter (see below)
9. Return `202 Accepted` with `document_id`

---

### `GET /documents`
List all documents (paginated) — the single shared pool, not scoped to the requesting user.

**Returns:** `document_id`, `display_name`, `status`, `created_at` for each document.

Used to look up the `document_id` to pass as `replaces_doc_id` when uploading a replacement.

---

### `GET /documents/{doc_id}`
Get status of a single document.

**Returns:** full status detail — current stage, `error_message` if failed, `retry_count`, timestamps.

---

### `POST /documents/{doc_id}/retry`
Retry ingestion for a failed document without re-uploading. **Retries from the very start** of the pipeline (scan → ... → index), not from the failed stage — there is no per-stage state to resume from without a task chain.

**Flow:**
1. Check `status=failed` and `retry_count < MAX_RETRIES` (else reject — see "Retry Limit" below)
2. Reset to `status=pending`, increment `retry_count`
3. Re-schedule the full ingestion task, subject to the same concurrency limiter as a fresh upload
4. Requires the file to already be in S3 (it was uploaded successfully before any ingestion stage could fail)

---

### `DELETE /documents/{doc_id}`
Remove a document and all its derived data.

**Flow:**
1. Check current status — block or force-cancel if actively processing
2. Look up all chunk keys for this document: `SELECT chunk_key FROM chunks WHERE doc_id = ?`
3. Batch-delete those keys from Amazon S3 Vectors via `DeleteVectors` (groups of up to 500 keys per call)
4. Delete the corresponding rows from `chunks`
5. Delete object from S3
6. Soft-delete (preferred) or hard-delete the PostgreSQL row

---

## Ingestion Execution Model (no Celery/broker)

Ingestion is triggered directly from the FastAPI process using `asyncio.create_task`, bounded by a global `asyncio.Semaphore` that caps how many ingestion pipelines can run concurrently, regardless of how many `/upload` or `/retry` calls come in.

```python
INGESTION_CONCURRENCY_LIMIT = 5   # tune based on embedding-API rate limits and CPU/IO capacity
ingestion_semaphore = asyncio.Semaphore(INGESTION_CONCURRENCY_LIMIT)

async def run_ingestion(document_id: str):
    async with ingestion_semaphore:
        try:
            await scan(document_id)
            await parse(document_id)
            await chunk(document_id)
            await embed(document_id)      # writes vectors to Amazon S3 Vectors via PutVectors,
                                            # incrementally, recording each chunk_key in the
                                            # `chunks` table as it's written
            await index_finalize(document_id)   # marks status=indexed
            await maybe_delete_replaced_doc(document_id)  # if this was a replace
        except Exception as e:
            await mark_failed(document_id, error_message=str(e))

# in the /upload handler, after creating the DB row:
asyncio.create_task(run_ingestion(document_id))
```

**What the semaphore controls:** requests beyond the concurrency limit simply wait in-process for a free slot — they are not rejected, just queued in memory until a running ingestion finishes. This is the direct mechanism for "control on how many ingestions run concurrently," and its limit should be tuned against the external embedding API's rate limits, since that's the most likely real constraint, not raw CPU.

**Retry-from-start rationale:** with no task chain and no per-stage persistence of intermediate output, the simplest correct behavior on failure is to re-run the entire pipeline for that `document_id`. `scan`/`parse`/`chunk` are cheap relative to `embed` (the external API call), so the added cost of redoing them on retry is acceptable for v1.

**Retry limit:** `retry_count` is tracked per document; after `MAX_RETRIES` (e.g. 3) consecutive failures, `/retry` refuses further automatic attempts and the document stays `status=failed` for manual investigation — prevents a permanently-broken file (corrupt PDF, unsupported format) from being retried indefinitely.

### Known limitation: no durability across process restarts

Because ingestion is an in-process `asyncio` task rather than a durable queue entry, **a server restart or crash mid-ingestion loses that in-flight task entirely** — nothing redelivers it. The document is left at whatever `status` it last reached (not necessarily `failed`; it may just be stuck at `pending`/an intermediate stage forever with no error recorded).

**Mitigation for v1:** a periodic sweep (a scheduled job or a cron-triggered internal endpoint) that finds documents with `status` not in a terminal state (`indexed`, `failed`, `deleted`, `duplicate`, `no_changes`) whose `updated_at` is older than a threshold (e.g. 10 minutes), and either auto-triggers `/retry` on them or flags them for manual attention. Without this sweep, a crash silently strands documents with no recovery path — this should not be skipped even though it's a small addition, since it's the only safety net given there's no durable queue underneath.

---

## Key Design Decisions

| Concern | Decision |
|---|---|
| Upload access | Exposed to authenticated end users, not backend-only. Rate-limited per-user via existing token bucket, reused on this endpoint |
| Document scoping | **None** — single shared/common document pool across all users. No `user_id` column on `documents` |
| Filename uniqueness | `display_name` unique across the entire pool (not per-user), enforced via a DB unique constraint filtered to non-deleted rows |
| Upload transport | Backend uploads directly to S3 (no presigned URL needed) |
| Blocking I/O | S3 upload and pipeline stages run as an `asyncio` task, not inline in the request handler — the handler returns `202` immediately after scheduling the task |
| Ingestion mechanism | In-process `asyncio.create_task`, bounded by a global `asyncio.Semaphore` — **no Celery, no RabbitMQ, no broker of any kind for v1** |
| Dedup | SHA-256 hash computed before S3 write; checked globally (no per-user scoping) against PostgreSQL, filtered to non-deleted documents |
| Update / replace semantics | Explicit `is_update` + `replaces_doc_id` — no auto-detection. Replacement always gets a fresh `document_id`; no version lineage tracked |
| Replace ordering | New document ingested and verified `indexed` before the old document is deleted |
| Retry strategy | Retry from the **start** of the pipeline, not from the failed stage — no per-stage task chain to resume from. Capped at `MAX_RETRIES` |
| Failure tracking | `status`, `error_message`, `retry_count` persisted per document |
| Deletion scope | Amazon S3 Vectors (batch-deleted by looked-up `chunk_key`s, since delete-by-filter isn't supported) + `chunks` rows + S3 raw object + PostgreSQL row |
| In-flight delete race | Ingestion checks for `status=deleted` before proceeding at each stage |
| Crash recovery | No durability by design (v1 tradeoff); mitigated by a periodic sweep for documents stuck in a non-terminal status |
| Persistence | PostgreSQL — `documents` (`document_id`, `display_name` (unique), `status`, `content_hash` (unique), `retry_count`, `error_message`, timestamps) + `chunks` (`chunk_key`, `doc_id`) for reliable vector cleanup |

---

## Document Status Lifecycle

```
pending → scanning → parsing → chunking → embedding → indexed
                                                    ↘ failed (retry_count < MAX → retryable)
                                                    ↘ deleted

duplicate / no_changes — terminal, short-circuited before "pending" is ever reached
```

---

## Amazon S3 Vectors — Specific Notes

- Vectors are organized into a **vector bucket → vector index** structure (analogous to Qdrant's collection); `doc_id` should be attached as **filterable metadata** on each vector so it can be used in query-time filters
- Writes are incremental via `PutVectors` — adding new vectors doesn't require touching or reprocessing existing ones, up to 500 vectors per call, with combined put+delete throughput capped at 1,000 requests/sec and 2,500 vectors/sec per index (GA limits) — comfortably enough for this system's scale
- **Deletion is key-based, not filter-based:** `DeleteVectors` requires an explicit list of vector `keys` (up to 500 per call) — there is no equivalent to Qdrant's "delete everything matching this filter." This is why the `chunks` table (above) exists: it's the only reliable way to know which keys to pass to `DeleteVectors` for a given `doc_id`
- Metadata filtering at **query time** (via `QueryVectors`) works well and is evaluated together with the similarity search itself, not as a separate post-filter step — this is fine for retrieval, but should not be relied on as a substitute for the `chunks` table when the goal is exhaustively enumerating a document's vectors (it's a top-K nearest-neighbor operation, not a full scan)
- Filterable metadata limits: up to 2KB per vector, up to 50 total metadata keys per vector (10 of which may be non-filterable)
- S3 Vectors provides **strong read-after-write consistency** — a query issued immediately after a write or delete reflects it, removing the eventual-consistency concern previously flagged for Qdrant's `wait=true` pattern in the replace flow
- **Latency profile differs from Qdrant:** S3 Vectors is positioned by AWS for large-scale, infrequent-access vector workloads, with query latency around 100ms for warm/frequent queries and sub-second for cold ones — workable for a chat-style RAG bot at moderate volume, but not built for sustained high-QPS real-time search the way Qdrant is. If query volume grows significantly, AWS's own recommended pattern is tiering: keep bulk/cold vector data in S3 Vectors and promote hot/frequently-queried data to Amazon OpenSearch Service for low-latency serving. Worth treating this as a scaling checkpoint, not an immediate concern at current volume

---

## Open Items for Next Pass
- Tuning `INGESTION_CONCURRENCY_LIMIT` against the embedding provider's actual rate limits
- Sweep job implementation for stuck/non-terminal documents (schedule, threshold, auto-retry vs. flag-only)
- **v2 candidate:** reintroduce Celery + a broker once ingestion volume or durability requirements outgrow the in-process model — the semaphore approach here is a deliberate v1 tradeoff, not a permanent architectural stance
- Chunk metadata schema beyond `doc_id` — additional fields to store per vector (page number, section, embedding model version, etc.), and correspondingly whether any of these also belong in the `chunks` table or only as S3 Vectors metadata
- Batch-size tuning for `chunks` lookups + `DeleteVectors` calls on very large documents (many chunks spanning multiple 500-key batches)