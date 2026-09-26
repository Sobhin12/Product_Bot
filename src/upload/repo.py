"""`documents` table access. Every function takes an autocommit connection."""
import psycopg
from psycopg.rows import dict_row

from . import config

COLUMNS = (
    "doc_id, display_name, content_hash, s3_key, status, retry_count, error_message, "
    "replaces_doc_id, created_at, updated_at"
)


def _one(conn: psycopg.Connection, sql: str, params: tuple) -> dict | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def get(conn: psycopg.Connection, doc_id: str) -> dict | None:
    return _one(conn, f"SELECT {COLUMNS} FROM documents WHERE doc_id = %s", (doc_id,))


def list_live(conn: psycopg.Connection, limit: int, offset: int) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"SELECT {COLUMNS} FROM documents WHERE status <> 'deleted' "
            "ORDER BY created_at DESC, doc_id LIMIT %s OFFSET %s",
            (limit, offset),
        )
        return cur.fetchall()


def by_name(conn: psycopg.Connection, display_name: str) -> dict | None:
    """The live document holding this name (a document mid-replace does not hold it)."""
    return _one(
        conn,
        f"SELECT {COLUMNS} FROM documents "
        "WHERE display_name = %s AND status <> 'deleted' AND replaces_doc_id IS NULL",
        (display_name,),
    )


def by_hash(conn: psycopg.Connection, content_hash: str) -> dict | None:
    return _one(
        conn, f"SELECT {COLUMNS} FROM documents WHERE content_hash = %s AND status <> 'deleted'", (content_hash,)
    )


def insert(
    conn: psycopg.Connection, doc_id: str, display_name: str, content_hash: str, s3_key: str, replaces: str | None
) -> None:
    conn.execute(
        "INSERT INTO documents (doc_id, display_name, content_hash, s3_key, status, replaces_doc_id) "
        "VALUES (%s, %s, %s, %s, 'pending', %s)",
        (doc_id, display_name, content_hash, s3_key, replaces),
    )


def set_status(conn: psycopg.Connection, doc_id: str, status: str) -> bool:
    """False if the document has been deleted meanwhile: the caller must stop."""
    cur = conn.execute(
        "UPDATE documents SET status = %s, updated_at = now() WHERE doc_id = %s AND status <> 'deleted'",
        (status, doc_id),
    )
    return cur.rowcount == 1


def mark_failed(conn: psycopg.Connection, doc_id: str, message: str) -> None:
    conn.execute(
        "UPDATE documents SET status = 'failed', error_message = %s, updated_at = now() "
        "WHERE doc_id = %s AND status <> 'deleted'",
        (message, doc_id),
    )


def mark_deleted(conn: psycopg.Connection, doc_id: str) -> None:
    conn.execute("UPDATE documents SET status = 'deleted', updated_at = now() WHERE doc_id = %s", (doc_id,))


def clear_replaces(conn: psycopg.Connection, doc_id: str) -> None:
    conn.execute("UPDATE documents SET replaces_doc_id = NULL, updated_at = now() WHERE doc_id = %s", (doc_id,))


def start_retry(conn: psycopg.Connection, doc_id: str) -> bool:
    """failed -> pending with retry_count + 1, only while under the retry limit."""
    cur = conn.execute(
        "UPDATE documents SET status = 'pending', retry_count = retry_count + 1, error_message = NULL, "
        "updated_at = now() WHERE doc_id = %s AND status = 'failed' AND retry_count < %s",
        (doc_id, config.MAX_RETRIES),
    )
    return cur.rowcount == 1


def stuck(conn: psycopg.Connection, older_than_seconds: int) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"SELECT {COLUMNS} FROM documents WHERE status = ANY(%s) "
            "AND updated_at < now() - make_interval(secs => %s)",
            (list(config.ACTIVE_STATUSES), older_than_seconds),
        )
        return cur.fetchall()
