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
            "SELECT doc_id FROM documents WHERE display_name LIKE %s ESCAPE E'\\\\'", (_like_prefix(policy),)
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
