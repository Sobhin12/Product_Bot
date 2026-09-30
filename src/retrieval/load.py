"""Load one ingested document (data/output/<stem>/{parents,children}.jsonl) into
Postgres and the vector store. The `documents` row must already exist.

    python -m retrieval.load <stem> --display-name "ReAssure 3.0 Policy Wordings"
"""
import argparse
import json
import uuid
from collections.abc import Callable
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from ingestion.split import count_tokens

from . import config, db
from .vectors import S3VectorStore, VectorStore


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_document(
    conn: psycopg.Connection,
    store: VectorStore,
    embed: Callable[[list[str]], list[list[float]]],
    doc_id: str,
    parents: list[dict],
    children: list[dict],
) -> None:
    """Write one document's parents, chunks and vectors. A failure leaves nothing
    behind, and re-running for the same doc_id replaces what an earlier attempt wrote."""
    parent_ids = {p["id"] for p in parents}
    orphans = [c["id"] for c in children if c["parent_id"] not in parent_ids]
    if orphans:
        raise ValueError(f"children without a parent: {orphans[:5]}")
    too_long = [c["id"] for c in children if count_tokens(c["text"]) > config.EMBED_MAX_TOKENS]
    if too_long:
        raise ValueError(f"children over the embedding limit: {too_long[:5]}")

    def ns(local_id: str) -> str:
        return f"{doc_id}:{local_id}"  # ids restart per file; namespace them by document

    written: list[str] = []
    try:
        with conn.transaction():
            conn.execute("DELETE FROM parents WHERE doc_id = %s", (doc_id,))  # cascades to chunks
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO parents (parent_id, doc_id, parent_text, title, section, clauses, source_pages) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    [
                        (ns(p["id"]), doc_id, p["text"], p.get("title"), p.get("section"),
                         Jsonb(p["clauses"]), Jsonb(p["source_pages"]))
                        for p in parents
                    ],
                )
                cur.executemany(
                    "INSERT INTO chunks (chunk_key, doc_id, parent_id, embedding_text, kind) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    [
                        (ns(c["id"]), doc_id, ns(c["parent_id"]), c["text"], c.get("kind", "clause"))
                        for c in children
                    ],
                )
            vectors = embed([c["text"] for c in children])
            items = [
                {
                    "key": ns(c["id"]),
                    "vector": v,
                    "metadata": {"doc_id": doc_id, "parent_id": ns(c["parent_id"]), "chunk_key": ns(c["id"])},
                }
                for c, v in zip(children, vectors)
            ]
            for i in range(0, len(items), config.PUT_BATCH):
                batch = items[i : i + config.PUT_BATCH]
                store.put(batch)
                written += [b["key"] for b in batch]
    except BaseException:
        if written:
            store.delete(written)
        raise


def main() -> None:
    from .embed import embed_documents

    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="folder name under data/output")
    ap.add_argument("--display-name", required=True)
    args = ap.parse_args()

    src = config.OUTPUT_DIR / args.stem
    parents, children = read_jsonl(src / "parents.jsonl"), read_jsonl(src / "children.jsonl")
    doc_id = str(uuid.uuid4())
    with db.connect(autocommit=True) as conn:
        db.init_schema(conn)
        conn.execute("INSERT INTO documents (doc_id, display_name) VALUES (%s, %s)", (doc_id, args.display_name))
        try:
            load_document(conn, S3VectorStore(), embed_documents, doc_id, parents, children)
        except BaseException:
            conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
            raise
    print(f"loaded {args.display_name!r} as {doc_id}: {len(parents)} parents, {len(children)} children")


if __name__ == "__main__":
    main()
