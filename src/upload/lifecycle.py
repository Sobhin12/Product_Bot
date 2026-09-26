"""Document lifecycle operations behind the endpoints: upload, retry, delete.

Each takes an autocommit connection and does only synchronous work, so the API can run
it in a thread. Scheduling the ingestion task is the caller's job."""
import hashlib
import logging
import uuid
from typing import BinaryIO

import psycopg

from retrieval.vectors import VectorStore

from . import config, repo
from .blob import BlobStore

log = logging.getLogger("upload")


class LifecycleError(Exception):
    """A request the API should refuse, with the HTTP status to use."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code, self.message = status_code, message


def s3_key(doc_id: str, display_name: str) -> str:
    return f"documents/{doc_id}/{display_name}"


def local_path(doc_id: str):
    return config.UPLOAD_DIR / f"{doc_id}.pdf"


def _validate_name(display_name: str) -> str:
    name = display_name.strip()
    if not name or len(name) > config.MAX_NAME_LENGTH:
        raise LifecycleError(422, f"display_name must be 1-{config.MAX_NAME_LENGTH} characters")
    if any(c in name for c in "/\\") or any(ord(c) < 32 for c in name):
        raise LifecycleError(422, "display_name must not contain slashes or control characters")
    return name


def _save_and_hash(stream: BinaryIO, dest) -> str:
    """Copy the upload to `dest`, enforcing the size limit and the PDF signature; returns SHA-256."""
    digest, size = hashlib.sha256(), 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(dest, "wb") as out:
            while chunk := stream.read(config.READ_CHUNK):
                if size == 0 and not chunk.startswith(b"%PDF-"):
                    raise LifecycleError(415, "file is not a PDF")
                size += len(chunk)
                if size > config.MAX_UPLOAD_BYTES:
                    raise LifecycleError(413, f"file exceeds {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
                digest.update(chunk)
                out.write(chunk)
        if size == 0:
            raise LifecycleError(422, "file is empty")
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def upload(
    conn: psycopg.Connection,
    stream: BinaryIO,
    filename: str,
    display_name: str,
    is_update: bool,
    replaces_doc_id: str | None,
) -> dict:
    """Validate, dedup and register an upload. Returns {"doc_id", "status"} where status is
    "accepted" (an ingestion must now be scheduled), "duplicate" or "no_changes"."""
    name = _validate_name(display_name)
    if not filename.lower().endswith(".pdf"):
        raise LifecycleError(415, "only .pdf files are accepted")
    if is_update and not replaces_doc_id:
        raise LifecycleError(422, "replaces_doc_id is required when is_update is true")
    if replaces_doc_id and not is_update:
        raise LifecycleError(422, "replaces_doc_id given but is_update is false")

    if is_update:
        target = _live_document(conn, replaces_doc_id)
        if target["status"] in config.ACTIVE_STATUSES:
            raise LifecycleError(409, "the document being replaced is still being processed")

    holder = repo.by_name(conn, name)
    if holder and not (is_update and str(holder["doc_id"]) == replaces_doc_id):
        raise LifecycleError(
            409, "a document with this name already exists: choose another name or replace the existing one"
        )

    doc_id = str(uuid.uuid4())
    path = local_path(doc_id)
    content_hash = _save_and_hash(stream, path)

    existing = repo.by_hash(conn, content_hash)
    if existing:
        path.unlink(missing_ok=True)
        unchanged = is_update and str(existing["doc_id"]) == replaces_doc_id
        return {"doc_id": str(existing["doc_id"]), "status": "no_changes" if unchanged else "duplicate"}

    try:
        repo.insert(conn, doc_id, name, content_hash, s3_key(doc_id, name), replaces_doc_id)
    except psycopg.errors.UniqueViolation:  # lost a race with a concurrent upload
        path.unlink(missing_ok=True)
        raise LifecycleError(409, "a document with this name or content was uploaded at the same time")
    return {"doc_id": doc_id, "status": "accepted"}


def _live_document(conn: psycopg.Connection, doc_id: str) -> dict:
    try:
        doc = repo.get(conn, doc_id)
    except psycopg.errors.InvalidTextRepresentation:  # not a UUID
        doc = None
    if doc is None or doc["status"] == "deleted":
        raise LifecycleError(404, "document not found")
    return doc


def get_document(conn: psycopg.Connection, doc_id: str) -> dict:
    return _live_document(conn, doc_id)


def start_retry(conn: psycopg.Connection, doc_id: str) -> None:
    """Move a failed document back to pending; the caller then schedules ingestion."""
    doc = _live_document(conn, doc_id)
    if doc["status"] != "failed":
        raise LifecycleError(409, f"only failed documents can be retried (status is {doc['status']})")
    if not repo.start_retry(conn, doc_id):
        raise LifecycleError(409, f"retry limit ({config.MAX_RETRIES}) reached: needs manual investigation")


def remove_document(conn: psycopg.Connection, doc_id: str, store: VectorStore, blobs: BlobStore) -> None:
    """Delete a document's vectors, rows, raw file and working copy, then soft-delete it.
    Vectors go first: `chunks` is the only record of their keys, so it must outlive them."""
    doc = _live_document(conn, doc_id)
    keys = [r[0] for r in conn.execute("SELECT chunk_key FROM chunks WHERE doc_id = %s", (doc_id,))]
    store.delete(keys)
    conn.execute("DELETE FROM parents WHERE doc_id = %s", (doc_id,))  # cascades to chunks
    if doc["s3_key"]:
        blobs.delete(doc["s3_key"])
    local_path(doc_id).unlink(missing_ok=True)
    repo.mark_deleted(conn, doc_id)


def delete_document(conn: psycopg.Connection, doc_id: str, store: VectorStore, blobs: BlobStore) -> None:
    doc = _live_document(conn, doc_id)
    if doc["status"] in config.ACTIVE_STATUSES:
        raise LifecycleError(409, "the document is still being processed: try again when it has finished")
    remove_document(conn, doc_id, store, blobs)
