"""Smoke tests for the Streamlit app against a stubbed API."""
import json
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "src" / "frontend" / "app.py")


class Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text or json.dumps(body)
        self.ok = status < 400
        self.headers = {"content-type": "application/json"}
        self.encoding = None

    def json(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_content(self, chunk_size=None, decode_unicode=False):
        yield from ["Clause ", "4.1 says ", "24 months."]


DOCS = [
    {"doc_id": "1", "display_name": "ReAssure 3.0 Policy Wordings", "status": "indexed", "retry_count": 0,
     "error_message": None, "created_at": "2026-09-26T10:00:00+00:00"},
    {"doc_id": "2", "display_name": "ReAssure 3.0 CIS", "status": "failed", "retry_count": 1,
     "error_message": "Failed during parsing: RuntimeError", "created_at": "2026-09-26T11:00:00+00:00"},
    {"doc_id": "3", "display_name": "Companion Wordings", "status": "indexed", "retry_count": 0,
     "error_message": None, "created_at": "2026-09-26T12:00:00+00:00"},
]


@pytest.fixture
def calls(monkeypatch):
    log = []

    def request(method, url, **kw):
        log.append((method, url, kw))
        if method == "GET" and url.endswith("/documents"):
            return Resp(200, DOCS)
        return Resp(202, {"doc_id": "9", "status": "pending"})

    def post(url, **kw):
        log.append(("POST", url, kw))
        return Resp(200, None, text="")

    monkeypatch.setattr(requests, "request", request)
    monkeypatch.setattr(requests, "post", post)
    return log


def test_policy_list_is_fixed_and_chat_waits_for_a_selection(calls):
    at = AppTest.from_file(APP, default_timeout=20).run()
    assert not at.exception
    assert list(at.radio(key="policy").options) == ["ReAssure 3.0"]  # not derived from the uploaded documents
    assert at.radio(key="policy").value is None
    assert at.chat_input[0].disabled and any("Select a policy" in i.value for i in at.info)


def test_choosing_a_policy_enables_chat_and_sends_it_as_the_prefix(calls):
    at = AppTest.from_file(APP, default_timeout=20).run()
    at.radio(key="policy").set_value("ReAssure 3.0").run()
    assert not at.exception and not at.chat_input[0].disabled

    at.chat_input[0].set_value("cataract waiting period?").run()
    assert not at.exception
    assert [m.markdown[0].value for m in at.chat_message] == ["cataract waiting period?", "Clause 4.1 says 24 months."]
    query = [c for c in calls if c[1].endswith("/query")][0]
    assert query[2]["json"] == {"query": "cataract waiting period?", "policy": "ReAssure 3.0"}


def test_chat_works_with_no_documents_uploaded(monkeypatch):
    monkeypatch.setattr(requests, "request", lambda *a, **k: Resp(200, []))
    monkeypatch.setattr(requests, "post", lambda url, **kw: Resp(200, None, text=""))
    at = AppTest.from_file(APP, default_timeout=20).run()
    at.radio(key="policy").set_value("ReAssure 3.0").run()
    at.chat_input[0].set_value("hi").run()
    assert not at.exception and len(at.chat_message) == 2


def test_api_down_shows_an_error_not_a_crash(monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError()

    monkeypatch.setattr(requests, "request", down)
    at = AppTest.from_file(APP, default_timeout=20).run()
    assert not at.exception and any("Cannot reach the API" in e.value for e in at.error)
