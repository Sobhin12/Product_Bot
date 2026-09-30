import hashlib
import uuid

import psycopg
import pytest

from retrieval import config, db
from retrieval.load import load_document
from retrieval.retrieve import add_policy, dedup_parents, list_policies, retrieve
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
        c = db.connect(autocommit=True)
    except psycopg.OperationalError:
        pytest.skip("Postgres not running (docker compose up -d)")
    db.init_schema(c)
    c.execute("DELETE FROM documents WHERE display_name LIKE 'TEST %'")
    c.execute("DELETE FROM policies WHERE name LIKE 'TEST %'")
    yield c
    c.execute("DELETE FROM documents WHERE display_name LIKE 'TEST %'")
    c.execute("DELETE FROM policies WHERE name LIKE 'TEST %'")
    c.close()


def make_doc(conn, name: str, status: str = "indexed") -> str:
    doc_id = str(uuid.uuid4())
    conn.execute("INSERT INTO documents (doc_id, display_name, status) VALUES (%s, %s, %s)", (doc_id, name, status))
    return doc_id


def load(conn, store, embed, name: str, children=CHILDREN) -> str:
    doc_id = make_doc(conn, name)
    load_document(conn, store, embed, doc_id, PARENTS, children)
    return doc_id


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
    doc_id = load(conn, store, fake_embed, "TEST ReAssure Wordings")

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
    load(conn, store, fake_embed, "TEST ReAssure Wordings")
    load(conn, store, fake_embed, "TEST Companion Wordings")

    got = retrieve(conn, store, fake_vec, "cataract", "TEST Companion", top_k=10)
    doc_ids = {str(r[0]) for r in conn.execute("SELECT doc_id FROM documents WHERE display_name LIKE 'TEST Companion%'")}
    assert got and all(p.parent_id.split(":")[0] in doc_ids for p in got)


def test_retrieve_reports_each_stage_timing(conn):
    store = InMemoryVectorStore()
    load(conn, store, fake_embed, "TEST ReAssure Wordings")
    timings: dict = {}
    assert retrieve(conn, store, fake_vec, "cataract", "TEST ReAssure", timings=timings)
    assert set(timings) == {"db_ms", "embed_ms", "search_ms"}
    assert all(isinstance(v, int) and v >= 0 for v in timings.values())


def test_retrieve_timings_for_an_unknown_policy_stop_at_the_document_lookup(conn):
    timings: dict = {}
    retrieve(conn, InMemoryVectorStore(), fake_vec, "anything", "TEST NoSuchPolicy", timings=timings)
    assert set(timings) == {"db_ms"}


def test_like_wildcards_in_policy_are_literal(conn):
    store = InMemoryVectorStore()
    load(conn, store, fake_embed, "TEST ReAssure Wordings")
    assert retrieve(conn, store, fake_vec, "cataract", "TEST %") == []


def test_failed_embedding_leaves_nothing_behind(conn):
    store = InMemoryVectorStore()

    def broken(texts):
        raise RuntimeError("endpoint down")

    doc_id = make_doc(conn, "TEST Broken")
    with pytest.raises(RuntimeError):
        load_document(conn, store, broken, doc_id, PARENTS, CHILDREN)
    assert store.items == {}
    assert conn.execute("SELECT count(*) FROM parents WHERE doc_id = %s", (doc_id,)).fetchone()[0] == 0


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
    doc_id = make_doc(conn, "TEST Flaky")
    with pytest.raises(RuntimeError):
        load_document(conn, store, fake_embed, doc_id, PARENTS, CHILDREN)
    assert store.items == {}
    assert conn.execute("SELECT count(*) FROM parents WHERE doc_id = %s", (doc_id,)).fetchone()[0] == 0


def test_reloading_the_same_document_replaces_instead_of_duplicating(conn):
    store = InMemoryVectorStore()
    doc_id = load(conn, store, fake_embed, "TEST Twice")
    load_document(conn, store, fake_embed, doc_id, PARENTS, CHILDREN)
    assert conn.execute("SELECT count(*) FROM chunks WHERE doc_id = %s", (doc_id,)).fetchone()[0] == 3
    assert len(store.items) == 3


def test_documents_that_are_not_indexed_are_not_searchable(conn):
    store = InMemoryVectorStore()
    doc_id = load(conn, store, fake_embed, "TEST Pending Doc")
    assert retrieve(conn, store, fake_vec, "cataract", "TEST Pending")
    conn.execute("UPDATE documents SET status = 'embedding' WHERE doc_id = %s", (doc_id,))
    assert retrieve(conn, store, fake_vec, "cataract", "TEST Pending") == []


def test_orphan_child_is_rejected_before_any_write(conn):
    bad = CHILDREN + [{"id": "C0009", "parent_id": "P9999", "text": "x", "kind": "clause"}]
    store = InMemoryVectorStore()
    with pytest.raises(ValueError, match="without a parent"):
        load_document(conn, store, fake_embed, make_doc(conn, "TEST Orphan"), PARENTS, bad)
    assert store.items == {}


def test_add_policy_then_list_returns_it_alphabetically(conn):
    assert add_policy(conn, "TEST Zeta") is True
    assert add_policy(conn, "TEST Alpha") is True
    names = list_policies(conn)
    assert names.index("TEST Alpha") < names.index("TEST Zeta")


def test_add_policy_twice_is_idempotent_and_reports_the_duplicate(conn):
    assert add_policy(conn, "TEST Dup") is True
    assert add_policy(conn, "TEST Dup") is False
    assert list_policies(conn).count("TEST Dup") == 1


def test_add_policy_trims_whitespace(conn):
    add_policy(conn, "  TEST Padded  ")
    assert "TEST Padded" in list_policies(conn)


@pytest.mark.parametrize("bad", ["", "   ", "x" * 101, "TEST bad\tname"])
def test_add_policy_rejects_invalid_names(conn, bad):
    with pytest.raises(ValueError):
        add_policy(conn, bad)


# ---- S3 Vectors retry -------------------------------------------------------

from retrieval import vectors as vectors_mod


class FakeAWSError(Exception):
    """Stand-in for a botocore connection-level error (ConnectionClosedError etc.)."""


def test_call_with_retry_retries_transient_failures_then_succeeds(monkeypatch):
    monkeypatch.setattr(vectors_mod, "_retryable", lambda e: isinstance(e, FakeAWSError))
    monkeypatch.setattr(vectors_mod.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeAWSError("connection closed")
        return "ok"

    assert vectors_mod.call_with_retry(flaky) == "ok" and calls["n"] == 3


def test_call_with_retry_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(vectors_mod, "_retryable", lambda e: isinstance(e, FakeAWSError))
    monkeypatch.setattr(vectors_mod.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise FakeAWSError("still down")

    with pytest.raises(FakeAWSError):
        vectors_mod.call_with_retry(always_fails, max_attempts=3)
    assert calls["n"] == 3


def test_call_with_retry_does_not_retry_non_retryable_errors(monkeypatch):
    monkeypatch.setattr(vectors_mod, "_retryable", lambda e: False)
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise ValueError("malformed filter")

    with pytest.raises(ValueError):
        vectors_mod.call_with_retry(bad_request)
    assert calls["n"] == 1


def test_is_retryable_classifies_connection_errors_and_throttling():
    from botocore.exceptions import (
        ClientError,
        ConnectionClosedError,
        ConnectTimeoutError,
        EndpointConnectionError,
        ReadTimeoutError,
        SSLError,
    )

    # botocore.exceptions.ConnectionError is the base of all of these; retrying on the
    # base (rather than listing each subclass) is the point of this test.
    for exc in (
        ConnectionClosedError(endpoint_url="https://x"),
        SSLError(endpoint_url="https://x", error="write interrupted"),
        ConnectTimeoutError(endpoint_url="https://x"),
        ReadTimeoutError(endpoint_url="https://x"),
        EndpointConnectionError(endpoint_url="https://x"),
    ):
        assert vectors_mod._retryable(exc), f"{type(exc).__name__} should be retryable"

    def client_error(code):
        return ClientError({"Error": {"Code": code}}, "PutVectors")

    assert vectors_mod._retryable(client_error("ThrottlingException"))
    assert not vectors_mod._retryable(client_error("ValidationException"))
    assert not vectors_mod._retryable(ValueError("not an aws error"))
