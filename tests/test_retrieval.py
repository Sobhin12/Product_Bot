import hashlib

import psycopg
import pytest

from retrieval import config, db
from retrieval.load import load_document
from retrieval.retrieve import dedup_parents, retrieve
from retrieval.vectors import Hit, InMemoryVectorStore

DIM = 16


def fake_vec(text: str) -> list[float]:
    """Deterministic bag-of-words vector: texts sharing words are close."""
    v = [0.0] * DIM
    for w in text.lower().split():
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    return v


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [fake_vec(t) for t in texts]


PARENTS = [
    {"id": "P0001", "text": "4.1 Cataract. Waiting period two years.", "title": "Cataract",
     "section": "4. Benefits", "clauses": ["4.1"], "source_pages": [10]},
    {"id": "P0002", "text": "5.1 Cancellation. Notice of seven days.", "title": "Cancellation",
     "section": "5. General", "clauses": ["5.1"], "source_pages": [20]},
]
CHILDREN = [
    {"id": "C0001", "parent_id": "P0001", "text": "cataract waiting period two years", "kind": "clause"},
    {"id": "C0002", "parent_id": "P0001", "text": "eye surgery cataract wait", "kind": "sidebar"},
    {"id": "C0003", "parent_id": "P0002", "text": "cancel policy notice seven days", "kind": "clause"},
]


@pytest.fixture
def conn():
    try:
        c = db.connect()
    except psycopg.OperationalError:
        pytest.skip("Postgres not running (docker compose up -d)")
    db.init_schema(c)
    c.execute("DELETE FROM documents WHERE display_name LIKE 'TEST %'")
    c.commit()
    yield c
    c.execute("DELETE FROM documents WHERE display_name LIKE 'TEST %'")
    c.commit()
    c.close()


def hit(key: str, parent: str, d: float = 0.0) -> Hit:
    return Hit(key, parent, d)


def test_dedup_keeps_first_occurrence_in_rank_order():
    hits = [hit("c1", "pB"), hit("c2", "pA"), hit("c3", "pB"), hit("c4", "pC"), hit("c5", "pA")]
    assert dedup_parents(hits) == ["pB", "pA", "pC"]


def test_dedup_empty():
    assert dedup_parents([]) == []


def test_unknown_policy_returns_empty_and_never_queries_vectors(conn):
    class Boom(InMemoryVectorStore):
        def query(self, *a, **k):
            raise AssertionError("QueryVectors must not be called with no doc_ids")

    assert retrieve(conn, Boom(), fake_vec, "anything", "TEST NoSuchPolicy") == []


def test_load_namespaces_ids_and_retrieve_round_trip(conn):
    store = InMemoryVectorStore()
    doc_id = load_document(conn, store, fake_embed, PARENTS, CHILDREN, "TEST ReAssure Wordings")

    assert all(k.startswith(f"{doc_id}:") for k in store.items)
    assert {i["metadata"]["chunk_key"] for i in store.items.values()} == set(store.items)
    assert conn.execute("SELECT count(*) FROM chunks WHERE doc_id = %s", (doc_id,)).fetchone()[0] == 3

    got = retrieve(conn, store, fake_vec, "cataract waiting period", "TEST ReAssure", top_k=3)
    assert got[0].text.startswith("4.1 Cataract")
    assert got[0].clauses == ["4.1"]
    # two children of P0001 match, but the parent is returned once
    assert [p.parent_id for p in got].count(f"{doc_id}:P0001") == 1


def test_retrieve_is_restricted_to_selected_policy(conn):
    store = InMemoryVectorStore()
    load_document(conn, store, fake_embed, PARENTS, CHILDREN, "TEST ReAssure Wordings")
    load_document(conn, store, fake_embed, PARENTS, CHILDREN, "TEST Companion Wordings")

    got = retrieve(conn, store, fake_vec, "cataract", "TEST Companion", top_k=10)
    doc_ids = {str(r[0]) for r in conn.execute("SELECT doc_id FROM documents WHERE display_name LIKE 'TEST Companion%'")}
    assert got and all(p.parent_id.split(":")[0] in doc_ids for p in got)


def test_like_wildcards_in_policy_are_literal(conn):
    store = InMemoryVectorStore()
    load_document(conn, store, fake_embed, PARENTS, CHILDREN, "TEST ReAssure Wordings")
    assert retrieve(conn, store, fake_vec, "cataract", "TEST %") == []


def test_failed_embedding_leaves_nothing_behind(conn):
    store = InMemoryVectorStore()

    def broken(texts):
        raise RuntimeError("endpoint down")

    with pytest.raises(RuntimeError):
        load_document(conn, store, broken, PARENTS, CHILDREN, "TEST Broken")
    assert store.items == {}
    assert conn.execute("SELECT count(*) FROM documents WHERE display_name = 'TEST Broken'").fetchone()[0] == 0


def test_failed_vector_put_deletes_earlier_batches(conn, monkeypatch):
    monkeypatch.setattr(config, "PUT_BATCH", 2)
    store = InMemoryVectorStore()
    real_put, calls = store.put, []

    def flaky(items):
        calls.append(len(items))
        if len(calls) == 2:
            raise RuntimeError("throttled")
        real_put(items)

    store.put = flaky
    with pytest.raises(RuntimeError):
        load_document(conn, store, fake_embed, PARENTS, CHILDREN, "TEST Flaky")
    assert store.items == {}
    assert conn.execute("SELECT count(*) FROM documents WHERE display_name = 'TEST Flaky'").fetchone()[0] == 0


def test_orphan_child_is_rejected_before_any_write(conn):
    bad = CHILDREN + [{"id": "C0009", "parent_id": "P9999", "text": "x", "kind": "clause"}]
    store = InMemoryVectorStore()
    with pytest.raises(ValueError, match="without a parent"):
        load_document(conn, store, fake_embed, PARENTS, bad, "TEST Orphan")
    assert store.items == {}
