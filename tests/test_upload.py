import hashlib
import io
import json
import time
import uuid

import psycopg
import pymupdf
import pytest
from fastapi.testclient import TestClient

from api.main import Services, create_app
from generation.answer import Generator
from ingestion import config as ingestion_config
from retrieval import db
from retrieval.retrieve import retrieve
from retrieval.vectors import InMemoryVectorStore
from tests.test_generation import FakeLLM
from tests.test_retrieval import CHILDREN, PARENTS, fake_embed, fake_vec
from upload import config, lifecycle, repo
from upload.blob import LocalBlobStore
from upload.pipeline import IngestionRunner

PREFIX = "TEST "


def make_pdf(text: str = "hello") -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text)
    return doc.tobytes()


class Env:
    """The app wired to fakes, with Postgres real."""

    def __init__(self, tmp_path):
        self.store = InMemoryVectorStore()
        self.blobs = LocalBlobStore(tmp_path / "blobs")
        self.parse_calls = 0
        self.fail_parse = False

        def parse(path, on_stage=lambda s: None):
            self.parse_calls += 1
            on_stage("parsing")
            if self.fail_parse:
                raise RuntimeError("docling blew up at C:\\secret\\path")
            on_stage("chunking")
            # Matches the real ingestion.run.process(): writes data/output/<doc_id>/report.md
            # as a side effect, so tests can exercise the pipeline's own archive-then-delete step.
            out = ingestion_config.OUTPUT_DIR / path.stem
            out.mkdir(parents=True, exist_ok=True)
            (out / "report.md").write_text(f"# report for {path.stem}\n", encoding="utf-8")
            return PARENTS, CHILDREN

        self.runner = IngestionRunner(self.blobs, self.store, fake_embed, parse=parse)
        self.conn = db.connect(autocommit=True)

        def answer_retrieve(q, policy, timings):
            with db.connect(autocommit=True) as c:
                return retrieve(c, self.store, fake_vec, q, policy, timings=timings)

        self.llm = FakeLLM(("The ", "answer"))
        self.services = Services(self.runner, Generator(self.llm, answer_retrieve), self.store, self.blobs)

    def upload(self, client, content: bytes, name: str, filename="policy.pdf", **form):
        data = {"display_name": PREFIX + name, **{k: str(v) for k, v in form.items()}}
        return client.post("/upload", files={"file": (filename, io.BytesIO(content), "application/pdf")}, data=data)

    def wait(self, client, doc_id, want=("indexed", "failed"), timeout=15):
        end = time.time() + timeout
        while time.time() < end:
            doc = client.get(f"/documents/{doc_id}")
            if doc.status_code == 404 or doc.json()["status"] in want:
                return doc
            time.sleep(0.05)
        raise AssertionError(f"timed out waiting for {doc_id}: {doc.json()}")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr("ingestion.config.OUTPUT_DIR", tmp_path / "output")
    try:
        e = Env(tmp_path)
    except psycopg.OperationalError:
        pytest.skip("Postgres not running (docker compose up -d)")
    db.init_schema(e.conn)
    e.conn.execute("DELETE FROM documents WHERE display_name LIKE 'TEST %'")
    e.conn.execute("DELETE FROM policies WHERE name LIKE 'TEST %'")
    yield e
    e.conn.execute("DELETE FROM documents WHERE display_name LIKE 'TEST %'")
    e.conn.execute("DELETE FROM policies WHERE name LIKE 'TEST %'")
    e.conn.close()


@pytest.fixture
def client(env):
    with TestClient(create_app(env.services)) as c:
        yield c


def test_upload_runs_the_whole_pipeline(env, client):
    content = make_pdf("policy one")
    r = env.upload(client, content, "Alpha Wordings")
    assert r.status_code == 202 and r.json()["status"] == "accepted"
    doc_id = r.json()["doc_id"]

    doc = env.wait(client, doc_id).json()
    assert doc["status"] == "indexed" and doc["error_message"] is None
    assert "content_hash" not in doc and "s3_key" not in doc

    stored = env.blobs.root / "documents" / doc_id / f"{PREFIX}Alpha Wordings"
    assert stored.read_bytes() == content  # raw file written to blob storage
    assert len(env.store.items) == 3 and all(k.startswith(doc_id) for k in env.store.items)
    assert env.conn.execute("SELECT count(*) FROM chunks WHERE doc_id = %s", (doc_id,)).fetchone()[0] == 3
    assert env.conn.execute(
        "SELECT content_hash FROM documents WHERE doc_id = %s", (doc_id,)
    ).fetchone()[0] == hashlib.sha256(content).hexdigest()
    assert not lifecycle.local_path(doc_id).exists()  # working copy cleaned up

    report = env.blobs.root / lifecycle.report_key(doc_id)
    assert report.exists() and report.read_text(encoding="utf-8") == f"# report for {doc_id}\n"
    assert not (ingestion_config.OUTPUT_DIR / doc_id).exists()  # local copy gone, but archived first


def test_status_moves_through_the_lifecycle(env, client):
    seen: list[str] = []
    real = repo.set_status

    def spy(conn, doc_id, status):
        seen.append(status)
        return real(conn, doc_id, status)

    import upload.pipeline as pipeline

    pipeline.repo.set_status = spy
    try:
        env.wait(client, env.upload(client, make_pdf("s"), "Stages").json()["doc_id"])
    finally:
        pipeline.repo.set_status = real
    assert seen == ["scanning", "parsing", "chunking", "embedding", "indexed"]


def test_same_content_under_another_name_is_a_duplicate(env, client):
    content = make_pdf("dup")
    first = env.upload(client, content, "Original").json()["doc_id"]
    env.wait(client, first)
    r = env.upload(client, content, "Copy")
    assert r.status_code == 200 and r.json() == {"doc_id": first, "status": "duplicate"}
    assert env.conn.execute("SELECT count(*) FROM documents WHERE display_name LIKE 'TEST %'").fetchone()[0] == 1
    assert env.parse_calls == 1


def test_duplicate_name_with_different_content_is_rejected(env, client):
    env.wait(client, env.upload(client, make_pdf("a"), "Same Name").json()["doc_id"])
    r = env.upload(client, make_pdf("b"), "Same Name")
    assert r.status_code == 409 and "already exists" in r.json()["detail"]


@pytest.mark.parametrize(
    "content,filename,name,code",
    [
        (b"not a pdf at all", "x.pdf", "Bad Magic", 415),
        (b"%PDF- fine", "x.txt", "Bad Ext", 415),
        (b"", "x.pdf", "Empty", 422),
        (b"%PDF-1.4", "x.pdf", "", 422),
        (b"%PDF-1.4", "x.pdf", "a/b", 422),
    ],
)
def test_invalid_uploads_are_refused_and_leave_nothing(env, client, content, filename, name, code):
    r = client.post(
        "/upload",
        files={"file": (filename, io.BytesIO(content), "application/pdf")},
        data={"display_name": (PREFIX + name) if name and "/" not in name else name},
    )
    assert r.status_code == code
    assert env.conn.execute("SELECT count(*) FROM documents WHERE display_name LIKE 'TEST %'").fetchone()[0] == 0
    assert not list(config.UPLOAD_DIR.glob("*")) if config.UPLOAD_DIR.exists() else True


def test_oversized_upload_is_refused(env, client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)
    r = env.upload(client, make_pdf("x" * 500), "Huge")
    assert r.status_code == 413
    assert not list(config.UPLOAD_DIR.glob("*"))


def test_unreadable_pdf_fails_the_scan_with_a_clear_message(env, client):
    r = env.upload(client, b"%PDF-1.4 garbage that is not really a pdf", "Corrupt")
    doc = env.wait(client, r.json()["doc_id"]).json()
    assert doc["status"] == "failed" and "not a readable PDF" in doc["error_message"]
    assert env.parse_calls == 0


def test_failure_is_recorded_without_leaking_internals_then_retry_succeeds(env, client):
    env.fail_parse = True
    doc_id = env.upload(client, make_pdf("f"), "Flaky").json()["doc_id"]
    doc = env.wait(client, doc_id).json()
    assert doc["status"] == "failed" and doc["retry_count"] == 0
    assert "parsing" in doc["error_message"] and "secret" not in doc["error_message"]
    assert env.store.items == {}

    env.fail_parse = False
    assert client.post(f"/documents/{doc_id}/retry").status_code == 202
    doc = env.wait(client, doc_id, want=("indexed",)).json()
    assert doc["status"] == "indexed" and doc["retry_count"] == 1 and doc["error_message"] is None


def test_retry_limit_and_state_checks(env, client, monkeypatch):
    monkeypatch.setattr(config, "MAX_RETRIES", 1)
    env.fail_parse = True
    doc_id = env.upload(client, make_pdf("r"), "Retries").json()["doc_id"]
    env.wait(client, doc_id)
    assert client.post(f"/documents/{doc_id}/retry").status_code == 202
    env.wait(client, doc_id)
    r = client.post(f"/documents/{doc_id}/retry")
    assert r.status_code == 409 and "retry limit" in r.json()["detail"]

    ok = env.upload(client, make_pdf("ok"), "Fine")
    env.fail_parse = False
    env.wait(client, ok.json()["doc_id"])
    assert client.post(f"/documents/{ok.json()['doc_id']}/retry").status_code == 409  # not failed


def test_retry_after_a_restart_refetches_the_file_from_blob_storage(env, client):
    env.fail_parse = True
    doc_id = env.upload(client, make_pdf("restart"), "Restart").json()["doc_id"]
    env.wait(client, doc_id)
    lifecycle.local_path(doc_id).unlink()  # the working copy is lost
    env.fail_parse = False
    client.post(f"/documents/{doc_id}/retry")
    assert env.wait(client, doc_id, want=("indexed",)).json()["status"] == "indexed"


def test_replace_swaps_content_and_removes_the_old_document(env, client):
    old = env.upload(client, make_pdf("v1"), "Policy").json()["doc_id"]
    env.wait(client, old)
    old_keys = set(env.store.items)

    r = env.upload(client, make_pdf("v2"), "Policy", is_update=True, replaces_doc_id=old)
    assert r.status_code == 202
    new = r.json()["doc_id"]
    assert env.wait(client, new, want=("indexed",)).json()["status"] == "indexed"

    deadline = time.time() + 5  # the old one is removed right after the new one is indexed
    while client.get(f"/documents/{old}").status_code != 404 and time.time() < deadline:
        time.sleep(0.05)
    assert client.get(f"/documents/{old}").status_code == 404
    assert not (old_keys & set(env.store.items)) and len(env.store.items) == 3
    assert not (env.blobs.root / "documents" / old).exists() or not list((env.blobs.root / "documents" / old).iterdir())
    names = [d["display_name"] for d in client.get("/documents").json() if d["display_name"].startswith(PREFIX)]
    assert names == [PREFIX + "Policy"]
    assert env.conn.execute("SELECT replaces_doc_id FROM documents WHERE doc_id = %s", (new,)).fetchone()[0] is None


def test_failed_replacement_leaves_the_old_document_untouched(env, client):
    old = env.upload(client, make_pdf("v1"), "Keep").json()["doc_id"]
    env.wait(client, old)
    env.fail_parse = True
    new = env.upload(client, make_pdf("v2"), "Keep", is_update=True, replaces_doc_id=old).json()["doc_id"]
    assert env.wait(client, new).json()["status"] == "failed"
    assert client.get(f"/documents/{old}").json()["status"] == "indexed"
    assert len(env.store.items) == 3


def test_replace_with_identical_content_is_no_changes(env, client):
    content = make_pdf("same")
    old = env.upload(client, content, "Unchanged").json()["doc_id"]
    env.wait(client, old)
    r = env.upload(client, content, "Unchanged", is_update=True, replaces_doc_id=old)
    assert r.json() == {"doc_id": old, "status": "no_changes"} and env.parse_calls == 1


def test_replace_argument_errors(env, client):
    assert env.upload(client, make_pdf("a"), "X", is_update=True).status_code == 422
    assert env.upload(client, make_pdf("b"), "Y", is_update=True, replaces_doc_id=str(uuid.uuid4())).status_code == 404
    assert env.upload(client, make_pdf("c"), "Z", is_update=True, replaces_doc_id="nonsense").status_code == 404
    assert env.upload(client, make_pdf("d"), "W", replaces_doc_id=str(uuid.uuid4())).status_code == 422


def test_delete_removes_everything_and_frees_the_name(env, client):
    content = make_pdf("del")
    doc_id = env.upload(client, content, "Doomed").json()["doc_id"]
    env.wait(client, doc_id)
    report = env.blobs.root / lifecycle.report_key(doc_id)
    assert report.exists()  # archived on success, per test_upload_runs_the_whole_pipeline
    assert client.delete(f"/documents/{doc_id}").status_code == 204

    assert not report.exists()  # deleting a document also removes its archived report
    assert env.store.items == {}
    assert env.conn.execute("SELECT count(*) FROM chunks WHERE doc_id = %s", (doc_id,)).fetchone()[0] == 0
    assert env.conn.execute("SELECT count(*) FROM parents WHERE doc_id = %s", (doc_id,)).fetchone()[0] == 0
    assert client.get(f"/documents/{doc_id}").status_code == 404
    assert client.delete(f"/documents/{doc_id}").status_code == 404
    again = env.upload(client, content, "Doomed")  # same name and same bytes are reusable
    assert again.status_code == 202 and again.json()["doc_id"] != doc_id


def test_deleting_a_document_with_no_archived_report_is_not_an_error(env, client):
    """A document uploaded before this feature existed has no report.md in blob storage;
    BlobStore.delete is a no-op on a missing key, so cleanup must not fail on it."""
    doc_id = env.upload(client, make_pdf("no report"), "NoReport").json()["doc_id"]
    env.wait(client, doc_id)
    (env.blobs.root / lifecycle.report_key(doc_id)).unlink()  # simulate: never archived
    assert client.delete(f"/documents/{doc_id}").status_code == 204


def test_a_failed_report_archive_does_not_fail_the_ingestion(env, client, monkeypatch):
    """blobs.put is also how the raw PDF is stored (the "storing" stage, before scanning
    even runs) - only the report.md put must fail here, or the whole pipeline never
    reaches "indexed" and this test would prove nothing about the archive step itself."""
    real_put = env.blobs.put

    def flaky_put(path, key):
        if key.endswith("/report.md"):
            raise RuntimeError("S3 is down")
        real_put(path, key)

    monkeypatch.setattr(env.blobs, "put", flaky_put)
    doc_id = env.upload(client, make_pdf("archive fails"), "ArchiveFails").json()["doc_id"]
    doc = env.wait(client, doc_id).json()
    assert doc["status"] == "indexed" and doc["error_message"] is None
    assert not (env.blobs.root / lifecycle.report_key(doc_id)).exists()  # the archive genuinely failed


def test_delete_is_refused_while_processing(env, client):
    doc_id = env.upload(client, make_pdf("busy"), "Busy").json()["doc_id"]
    env.wait(client, doc_id)
    env.conn.execute("UPDATE documents SET status = 'embedding' WHERE doc_id = %s", (doc_id,))
    assert client.delete(f"/documents/{doc_id}").status_code == 409
    assert len(env.store.items) == 3


def test_list_is_paginated_and_hides_deleted(env, client):
    ids = [env.upload(client, make_pdf(f"doc {i}"), f"List {i}").json()["doc_id"] for i in range(3)]
    for i in ids:
        env.wait(client, i)
    client.delete(f"/documents/{ids[0]}")
    mine = lambda docs: [d for d in docs if d["display_name"].startswith(PREFIX)]
    assert {d["doc_id"] for d in mine(client.get("/documents?limit=50").json())} == set(ids[1:])
    assert len(client.get("/documents?limit=1&offset=0").json()) == 1
    assert client.get("/documents?limit=0").status_code == 422


def test_concurrent_uploads_respect_the_ingestion_semaphore(env, tmp_path):
    active, peak = 0, 0
    real = env.runner.parse

    def slow(path, on_stage=lambda s: None):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        time.sleep(0.15)
        try:
            return real(path, on_stage=on_stage)
        finally:
            active -= 1

    runner = IngestionRunner(env.blobs, env.store, fake_embed, parse=slow, concurrency=2)
    env.services.runner = runner
    with TestClient(create_app(env.services)) as client:
        ids = [env.upload(client, make_pdf(f"c{i}"), f"Conc {i}").json()["doc_id"] for i in range(6)]
        for i in ids:
            assert env.wait(client, i).json()["status"] == "indexed"
    assert peak == 2


def test_sweep_restarts_documents_stranded_by_a_crash(env, client, monkeypatch):
    doc_id = env.upload(client, make_pdf("crash"), "Stranded").json()["doc_id"]
    env.wait(client, doc_id)
    env.conn.execute(
        "UPDATE documents SET status = 'parsing', updated_at = now() - interval '1 hour' WHERE doc_id = %s", (doc_id,)
    )
    env.store.items.clear()
    # its working copy died with the process; the sweep must refetch from blob storage
    assert (env.blobs.root / "documents" / doc_id).exists()

    async def sweep():
        revived = await env.runner.sweep_once()
        await env.runner.wait_idle()
        return revived

    import asyncio

    assert asyncio.run(sweep()) == [doc_id]
    doc = client.get(f"/documents/{doc_id}").json()
    assert doc["status"] == "indexed" and doc["retry_count"] == 1
    assert len(env.store.items) == 3


def test_sweep_leaves_running_and_recent_documents_alone(env, client):
    doc_id = env.upload(client, make_pdf("slow"), "Slow").json()["doc_id"]
    env.wait(client, doc_id)
    env.conn.execute("UPDATE documents SET status = 'embedding' WHERE doc_id = %s", (doc_id,))  # recent
    import asyncio

    assert asyncio.run(env.runner.sweep_once()) == []
    env.conn.execute("UPDATE documents SET updated_at = now() - interval '1 hour' WHERE doc_id = %s", (doc_id,))
    env.runner.running.add(doc_id)  # old, but this process is still working on it
    assert asyncio.run(env.runner.sweep_once()) == []


def ndjson(r) -> list[dict]:
    return [json.loads(line) for line in r.text.splitlines() if line]


def test_query_streams_ndjson_tokens_then_a_done_event(env, client):
    doc_id = env.upload(client, make_pdf("q"), "Queryable").json()["doc_id"]
    env.wait(client, doc_id)
    r = client.post("/query", json={"query": "cataract waiting period", "policy": PREFIX + "Queryable"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/x-ndjson"
    events = ndjson(r)
    assert "".join(e["text"] for e in events[:-1]) == "The answer"
    done = events[-1]
    assert done["type"] == "done" and done["input_tokens"] is None  # FakeLLM here reports no usage
    assert isinstance(done["latency_ms"], int)
    assert done["parents"] and done["parents"][0]["kept"] is True
    assert "Cataract" in env.llm.prompts[0][1]

    r = client.post("/query", json={"query": "cataract", "policy": PREFIX + "Nothing Like This"})
    events = ndjson(r)
    assert events[0]["text"] == "I couldn't find relevant information in the selected policy documents."
    assert events[-1]["parents"] == []


def test_query_validates_input(client):
    assert client.post("/query", json={"query": "", "policy": "P"}).status_code == 422
    assert client.post("/query", json={"query": "q"}).status_code == 422
    assert client.get("/health").json() == {"status": "ok"}


def test_policies_endpoint_lists_and_adds(env, client):
    before = client.get("/policies").json()
    assert isinstance(before, list)

    r = client.post("/policies", json={"name": "TEST Newco"})
    assert r.status_code == 201 and "TEST Newco" in r.json()
    assert "TEST Newco" in client.get("/policies").json()

    dup = client.post("/policies", json={"name": "TEST Newco"})
    assert dup.status_code == 409

    assert client.post("/policies", json={"name": "  "}).status_code == 422
    assert client.post("/policies", json={"name": "x" * 101}).status_code == 422
