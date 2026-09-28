"""Query-time semantic retrieval: policy pre-filter -> vector search -> parent dedup -> parent fetch."""
from collections.abc import Callable
from dataclasses import dataclass

import psycopg

from . import config
from .vectors import Hit, VectorStore


@dataclass
class Parent:
    parent_id: str
    text: str
    title: str | None
    section: str | None
    clauses: list[str]
    source_pages: list[int]


def _like_prefix(prefix: str) -> str:
    return prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def dedup_parents(hits: list[Hit]) -> list[str]:
    """Distinct parent_ids in rank order; the first (best) hit of a parent wins."""
    return list(dict.fromkeys(h.parent_id for h in hits))


MAX_POLICY_NAME_LENGTH = 100


def list_policies(conn: psycopg.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT name FROM policies ORDER BY name")]


def add_policy(conn: psycopg.Connection, name: str) -> bool:
    """Register a policy prefix for the chat picker. False if it already exists.
    Raises ValueError on an empty, overlong or control-character name."""
    name = name.strip()
    if not name or len(name) > MAX_POLICY_NAME_LENGTH:
        raise ValueError(f"name must be 1-{MAX_POLICY_NAME_LENGTH} characters")
    if any(ord(c) < 32 for c in name):
        raise ValueError("name must not contain control characters")
    cur = conn.execute("INSERT INTO policies (name) VALUES (%s) ON CONFLICT DO NOTHING", (name,))
    return cur.rowcount == 1


def retrieve(
    conn: psycopg.Connection,
    store: VectorStore,
    embed_query: Callable[[str], list[float]],
    query: str,
    policy: str,
    top_k: int = config.TOP_K,
) -> list[Parent]:
    """Parents for the LLM context, best first. Empty when no document matches `policy`."""
    doc_ids = [
        str(r[0])
        for r in conn.execute(
            "SELECT doc_id FROM documents WHERE status = 'indexed' AND display_name LIKE %s ESCAPE E'\\\\'",
            (_like_prefix(policy),),
        )
    ]
    if not doc_ids:
        return []  # QueryVectors rejects an empty $in list

    parent_ids = dedup_parents(store.query(embed_query(query), doc_ids, top_k))
    if not parent_ids:
        return []
    rows = {
        r[0]: r
        for r in conn.execute(
            "SELECT parent_id, parent_text, title, section, clauses, source_pages "
            "FROM parents WHERE parent_id = ANY(%s)",
            (parent_ids,),
        )
    }
    return [Parent(*rows[p]) for p in parent_ids]
