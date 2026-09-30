"""Query-time semantic retrieval: policy pre-filter -> vector search -> parent dedup -> parent fetch."""
import time
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


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def retrieve(
    conn: psycopg.Connection,
    store: VectorStore,
    embed_query: Callable[[str], list[float]],
    query: str,
    policy: str,
    top_k: int = config.TOP_K,
    timings: dict | None = None,
) -> list[Parent]:
    """Parents for the LLM context, best first. Empty when no document matches `policy`.
    `timings`, if given, is filled in place with the stages that ran: "db_ms" (document
    lookup plus parent fetch), "embed_ms" and "search_ms"."""
    timings = {} if timings is None else timings
    start = time.perf_counter()
    doc_ids = [
        str(r[0])
        for r in conn.execute(
            "SELECT doc_id FROM documents WHERE status = 'indexed' AND display_name LIKE %s ESCAPE E'\\\\'",
            (_like_prefix(policy),),
        )
    ]
    timings["db_ms"] = _ms(start)
    if not doc_ids:
        return []  # QueryVectors rejects an empty $in list

    start = time.perf_counter()
    vector = embed_query(query)
    timings["embed_ms"] = _ms(start)
    start = time.perf_counter()
    hits = store.query(vector, doc_ids, top_k)
    timings["search_ms"] = _ms(start)
    parent_ids = dedup_parents(hits)
    if not parent_ids:
        return []
    start = time.perf_counter()
    rows = {
        r[0]: r
        for r in conn.execute(
            "SELECT parent_id, parent_text, title, section, clauses, source_pages "
            "FROM parents WHERE parent_id = ANY(%s)",
            (parent_ids,),
        )
    }
    timings["db_ms"] += _ms(start)
    return [Parent(*rows[p]) for p in parent_ids]
