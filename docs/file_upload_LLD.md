# File Upload — Low Level Design

## Overview
Backend-triggered file upload for a production-grade RAG system. Single service (FastAPI) owns validation, dedup, S3 upload, and orchestration of the async ingestion pipeline via Celery. Persistence is PostgreSQL; vector storage is Qdrant.

---

## Endpoints

### `POST /upload`
Uploads a document and kicks off ingestion. Caller explicitly declares intent — new document vs. replacement of an existing one — no auto-detection.

**Request fields:**
- `file` — the document bytes
- `is_update` (bool) — whether this replaces an existing document
- `replaces_doc_id` (required if `is_update=true`) — the existing `document_id` being replaced

**Flow:**
1. Validate request — file size limit, content-type/extension allowlist (reject before touching disk)
2. Save file to temp local disk path
3. Compute file hash (SHA-256) — **before any S3 write**
4. Check hash against PostgreSQL, filtered to `status='indexed'` documents only (a deleted document's hash must not block re-upload)
   - **If duplicate found:**
     - If `is_update=false` → return the existing `document_id`, `status=duplicate`, skip entirely — no S3 write, no DB row, no Celery task
     - If `is_update=true` and the hash matches `replaces_doc_id` itself (content unchanged) → return `status=no_changes`, skip entirely — do not run a pointless delete + re-ingest
   - **If no duplicate** → proceed
5. Create DB row → fresh `document_id`, `status=pending` (this is always a **new** `document_id`, even on replace — no version lineage is tracked; replace is delete-old + insert-new, not an update-in-place)
6. Enqueue Celery task: `upload_to_s3_and_ingest(document_id, temp_path, is_update, replaces_doc_id)`
7. Return `202 Accepted` with `document_id`

---

### `GET /documents`
List all documents (paginated).

**Returns:** `document_id`, `filename`, `status`, `content_hash`, `created_at` for each document.

Used by the caller to look up the `document_id` to pass as `replaces_doc_id` when uploading an update — this endpoint is what makes "which existing doc is being replaced" answerable in practice.

---

### `GET /documents/{doc_id}`
Get status of a single document.

**Returns:** full status detail — current stage, `error_message` if failed, `failed_stage`, timestamps per stage (optional).

---

### `POST /documents/{doc_id}/retry`
Retry ingestion for a failed document without re-uploading.

**Flow:**
- Re-enqueue the Celery task starting from the last failed stage
- Requires file already in S3 (or temp path, if failure occurred pre-S3-upload)

---

### `DELETE /documents/{doc_id}`
Remove a document and all its derived data. Used both for direct user-initiated deletion and internally by the replace flow (see below) to clean up the old document after the new one is confirmed indexed.

**Flow:**
1. Check current status — block or force-cancel if actively processing
2. Delete chunks from Qdrant, filtered by `doc_id` payload field, with `wait=true` so the delete is confirmed complete before returning (avoids a window where the old content is still returned by search due to eventual consistency)
3. Delete object from S3
4. Soft-delete (preferred) or hard-delete the PostgreSQL row

---

## Celery Task: `upload_to_s3_and_ingest(document_id, temp_path, is_update, replaces_doc_id)`

1. Check DB status — abort if already `uploaded` or `deleted` (idempotency guard against duplicate task delivery)
2. Upload `temp_path` to S3
3. Delete `temp_path` from local disk
4. Update `status=uploaded`
5. Chain to ingestion task(s): `scanning → parsing → chunking → embedding → indexed`
6. **On successful reach of `indexed`:**
   - If `is_update=true` → call the delete flow (as in `DELETE /documents/{doc_id}`) against `replaces_doc_id` — the **old** document's chunks and DB row are only removed now, after the new document is confirmed indexed
7. **On failure at any stage:** leave `replaces_doc_id` (the old document) completely untouched — the old, still-valid version remains searchable while the new one is fixed/retried. A short window of duplicate content on success is preferred over a window with neither version present on failure.

---

## Key Design Decisions

| Concern | Decision |
|---|---|
| Upload transport | Backend uploads directly to S3 (no presigned URL needed — no untrusted client) |
| Blocking I/O | Never do S3 upload inline in the FastAPI request handler — always via Celery |
| Dedup | SHA-256 hash computed synchronously in `/upload`, **before** S3 write and before Celery enqueue; checked against PostgreSQL `content_hash` (indexed column), filtered to currently-`indexed` documents only |
| Update / replace semantics | Explicit `is_update` + `replaces_doc_id` flags from caller — no auto-detection. Replacement always gets a **fresh `document_id`**; no version lineage tracked |
| Replace ordering | New document is ingested and verified `indexed` **before** the old document is deleted — never delete-then-ingest, to avoid a zero-version failure window |
| Identical-content short-circuit | If an `is_update=true` upload's hash matches `replaces_doc_id`'s current hash, skip the whole flow (`status=no_changes`) rather than delete + re-ingest for no reason |
| Idempotency | Celery task checks DB status before acting, to survive at-least-once delivery |
| Failure tracking | `status`, `error_message`, `failed_stage` persisted per document |
| Deletion scope | Must clean up Qdrant chunks (by `doc_id`, indexed payload field, `wait=true`) **and** S3 object **and** PostgreSQL row — not just S3 |
| In-flight delete race | Each ingestion stage should check for `deleted` status before proceeding |
| Temp file cleanup | Deleted immediately after successful S3 upload; orphan cleanup job for failures |
| Persistence | PostgreSQL — `document_id`, `filename`, `status`, `content_hash` (indexed), `failed_stage`, `error_message`, timestamps |

---

## Document Status Lifecycle

```
pending → uploaded → scanning → parsing → chunking → embedding → indexed
                                                              ↘ failed (any stage)
                                                              ↘ deleted

duplicate / no_changes — terminal, short-circuited before "pending" is ever reached
```

---

## Qdrant-Specific Notes

- `doc_id` must be an **indexed payload field** on the collection (created at collection setup) — filter-based deletes and future metadata-filtered retrieval both depend on this; without it, filter operations fall back to a full scan as the corpus grows
- Chunk writes during ingestion use `upsert` — new chunks are added incrementally without touching existing collection data
- Deletes use `delete_points_by_filter` on `doc_id`, called with `wait=true` in the replace flow specifically, since the next step (new document already live) means both versions must not coexist any longer than the ingestion-verification window

---

## Open Items for Next Pass
- Ingestion task chain design: single monolithic task vs. per-stage chained tasks with independent retries
- Chunk-to-`doc_id` payload schema — exact fields stored alongside each vector (page number, section, embedding model version, etc.)