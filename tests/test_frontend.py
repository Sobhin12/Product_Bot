"""Smoke tests for the Streamlit app against a stubbed API."""
import json
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "src" / "frontend" / "app.py")


DONE_EVENT = {
    "type": "done", "input_tokens": 41, "output_tokens": 4, "latency_ms": 1234,
    "timings": {"embed_ms": 310, "search_ms": 120, "db_ms": 20, "retrieval_ms": 470,
                "llm_wait_ms": 0, "ttft_ms": 900, "generation_ms": 14200},
    "parents":[{"parent_id": "P1", "title": "Cataract", "section": "4. Benefits",
                 "clauses": ["4.1"], "source_pages": [10], "text": "4.1 Cataract. Waiting period 24 months.",
                 "kept": True}],
}
ANSWER_EVENTS = [
    {"type": "token", "text": "Clause "}, {"type": "token", "text": "4.1 says "},
    {"type": "token", "text": "24 months."}, DONE_EVENT,
]


class Resp:
    def __init__(self, status=200, body=None, text="", lines=None):
        self.status_code, self._body, self.text = status, body, text or json.dumps(body)
        self.ok = status < 400
        self.headers = {"content-type": "application/json"}
        self.encoding = None
        self._lines = lines if lines is not None else [json.dumps(e) for e in ANSWER_EVENTS]

    def json(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_lines(self, decode_unicode=False):
        yield from self._lines


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
        if method == "GET" and url.endswith("/policies"):
            return Resp(200, ["ReAssure 3.0"])
        return Resp(202, {"doc_id": "9", "status": "pending"})

    def post(url, **kw):
        log.append(("POST", url, kw))
        return Resp(200, None, text="")

    monkeypatch.setattr(requests, "request", request)
    monkeypatch.setattr(requests, "post", post)
    return log


def test_policy_list_comes_from_the_api_and_chat_waits_for_a_selection(calls):
    at = AppTest.from_file(APP, default_timeout=20).run()
    assert not at.exception
    assert list(at.radio(key="policy").options) == ["ReAssure 3.0"]  # from GET /policies, not the documents list
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


def parent_expanders(at):
    return [e for e in at.expander if e.label != "Manage a document"]


def test_answer_shows_token_counts_timing_and_the_parent_actually_used(calls):
    at = AppTest.from_file(APP, default_timeout=20).run()
    at.radio(key="policy").set_value("ReAssure 3.0").run()
    at.chat_input[0].set_value("cataract waiting period?").run()
    assert not at.exception

    stats = [c.value for c in at.caption if "Input tokens" in c.value][0]
    assert "Input tokens: 41" in stats and "Output tokens: 4" in stats and "1.2s" in stats
    stages = [c.value for c in at.caption if c.value.startswith("Stages:")][0]
    assert stages == (
        "Stages: embed 0.31s · vector search 0.12s · db 0.02s · retrieval total 0.47s · "
        "LLM queue 0.00s · first token 0.90s · generation 14.20s"
    )


def test_answer_without_timings_shows_no_stage_line(monkeypatch, calls):
    done = {k: v for k, v in DONE_EVENT.items() if k != "timings"}  # an API from before timings existed
    events = [{"type": "token", "text": "answer"}, done]
    monkeypatch.setattr(requests, "post", lambda url, **kw: Resp(200, None, lines=[json.dumps(e) for e in events]))

    at = AppTest.from_file(APP, default_timeout=20).run()
    at.radio(key="policy").set_value("ReAssure 3.0").run()
    at.chat_input[0].set_value("q").run()
    assert not at.exception
    assert any("Input tokens: 41" in c.value for c in at.caption)
    assert not any(c.value.startswith("Stages:") for c in at.caption)

    exps = parent_expanders(at)
    assert len(exps) == 1
    exp = exps[0]
    assert exp.label == "4.1 · p10 · Cataract" and "not sent" not in exp.label
    assert exp.text[0].value == "4.1 Cataract. Waiting period 24 months."


def test_dropped_parent_is_labeled_as_not_sent_to_the_model(monkeypatch, calls):
    done = {**DONE_EVENT, "parents": [
        {**DONE_EVENT["parents"][0], "kept": True},
        {"parent_id": "P2", "title": "General Terms", "section": "6", "clauses": ["6.1"],
         "source_pages": [34], "text": "huge clause", "kept": False},
    ]}
    events = [{"type": "token", "text": "answer"}, done]
    monkeypatch.setattr(requests, "post", lambda url, **kw: Resp(200, None, lines=[json.dumps(e) for e in events]))

    at = AppTest.from_file(APP, default_timeout=20).run()
    at.radio(key="policy").set_value("ReAssure 3.0").run()
    at.chat_input[0].set_value("q").run()
    assert not at.exception

    labels = {e.label for e in parent_expanders(at)}
    assert any("6.1" in l and "not sent" in l for l in labels)
    assert any("4.1" in l and "not sent" not in l for l in labels)


def test_only_the_latest_answer_shows_metadata(calls):
    at = AppTest.from_file(APP, default_timeout=20).run()
    at.radio(key="policy").set_value("ReAssure 3.0").run()
    at.chat_input[0].set_value("first question").run()
    assert len(parent_expanders(at)) == 1

    at.chat_input[0].set_value("second question").run()
    assert not at.exception
    # still exactly one metadata block, attached to the newest answer, not accumulated per turn
    assert len(parent_expanders(at)) == 1
    assert len([c for c in at.caption if "Input tokens" in c.value]) == 1


def test_switching_policy_clears_the_previous_answers_metadata(calls):
    at = AppTest.from_file(APP, default_timeout=20).run()
    at.radio(key="policy").set_value("ReAssure 3.0").run()
    at.chat_input[0].set_value("q").run()
    assert len(parent_expanders(at)) == 1

    at.radio(key="policy").set_value(None).run()
    assert not at.exception and len(parent_expanders(at)) == 0
    assert not any("Input tokens" in c.value for c in at.caption)


def test_chat_works_with_no_documents_uploaded(calls):
    """Documents and policies are independent: a policy can be selected with no
    documents uploaded yet (it just won't find anything to answer from)."""
    at = AppTest.from_file(APP, default_timeout=20).run()
    at.radio(key="policy").set_value("ReAssure 3.0").run()
    at.chat_input[0].set_value("hi").run()
    assert not at.exception and len(at.chat_message) == 2


def test_no_policies_registered_shows_guidance_instead_of_chat(monkeypatch):
    monkeypatch.setattr(requests, "request", lambda method, url, **k: Resp(200, []))
    at = AppTest.from_file(APP, default_timeout=20).run()
    assert not at.exception
    assert not at.chat_input and any("No policies are registered" in i.value for i in at.info)


def test_api_down_shows_an_error_not_a_crash(monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError()

    monkeypatch.setattr(requests, "request", down)
    at = AppTest.from_file(APP, default_timeout=20).run()
    assert not at.exception and any("Cannot reach the API" in e.value for e in at.error)
