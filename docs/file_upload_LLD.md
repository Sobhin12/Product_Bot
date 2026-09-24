# File Upload — Low Level Design

## Overview
Backend-triggered file upload for a production-grade RAG system. Single service (FastAPI) owns validation, S3 upload, dedup, and orchestration of the async ingestion pipeline via Celery.

---

## Endpoints

### `POST /upload`
Uploads a new document and kicks off ingestion.

**Flow:**
1. Validate request — file size limit, content-type/extension allowlist (reject before touching disk)
2. Save file to temp local disk path
3. Compute file hash (SHA-256)
4. Check hash against DB for duplicates
   - If duplicate → return existing `document_id`, `status=duplicate`, skip Celery entirely
5. Create DB row → `status=pending`
6. Enqueue Celery task: `upload_to_s3_and_ingest(document_id, temp_path)`
7. Return `202 Accepted` with `document_id`

---

### `GET /documents`
List all documents (paginated).

**Returns:** `document_id`, `filename`, `status`, `created_at` for each document.

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
Remove a document and all its derived data.

**Flow:**
1. Check current status — block or force-cancel if actively processing
2. Delete object from S3
3. Delete associated chunks/embeddings from the vector store (by `document_id`)
4. Soft-delete (preferred) or hard-delete the DB row

---

## Celery Task: `upload_to_s3_and_ingest(document_id, temp_path)`

1. Check DB status — abort if already `uploaded` or `deleted` (idempotency guard against duplicate task delivery)
2. Upload `temp_path` to S3
3. Delete `temp_path` from local disk
4. Update `status=uploaded`
5. Chain to ingestion task(s): `scanning → parsing → chunking → embedding → indexed`

---

## Key Design Decisions

| Concern | Decision |
|---|---|
| Upload transport | Backend uploads directly to S3 (no presigned URL needed — no untrusted client) |
| Blocking I/O | Never do S3 upload inline in the FastAPI request handler — always via Celery |
| Dedup | Hash computed synchronously in `/upload` endpoint, before enqueueing Celery |
| Idempotency | Celery task checks DB status before acting, to survive at-least-once delivery |
| Failure tracking | `status`, `error_message`, `failed_stage` persisted per document |
| Deletion | Must clean up S3 **and** vector store **and** DB row — not just S3 |
| In-flight delete race | Each ingestion stage should check for `deleted` status before proceeding |
| Temp file cleanup | Deleted immediately after successful S3 upload; orphan cleanup job for failures |

---

## Document Status Lifecycle

```
pending → uploaded → scanning → parsing → chunking → embedding → indexed
                                                              ↘ failed (any stage)
                                                              ↘ deleted
```

---

## Open Items for Next Pass
- Ingestion task chain design: single monolithic task vs. per-stage chained tasks with independent retries
- Vector store deletion mechanics: document → chunk mapping structure for efficient cleanup on `DELETE`