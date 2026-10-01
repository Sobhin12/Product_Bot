import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "scripts"))
from collect import collect, load_golden, parse_stream  # noqa: E402
from score import scorable, summarize  # noqa: E402


def golden(id_, policy, qtype="factual"):
    return {"id": id_, "policy": policy, "question": f"q {id_}", "answer": f"a {id_}",
            "relevant_context": [f"c {id_}"], "source": {"file": "x.pdf", "pages": [1]}, "question_type": qtype}


def done(parents, input_tokens=100, output_tokens=20):
    return json.dumps({"type": "done", "input_tokens": input_tokens, "output_tokens": output_tokens,
                       "latency_ms": 900, "timings": {"embed_ms": 300}, "parents": parents})


def parent(text, kept):
    return {"parent_id": text, "title": "", "section": "", "clauses": [], "source_pages": [3], "text": text, "kept": kept}


def test_load_golden_filters_by_policy_and_takes_the_first_n_of_each_file(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([golden("A1", "P1"), golden("A2", "P1"), golden("A3", "P1")]))
    (tmp_path / "b.json").write_text(json.dumps([golden("B1", "P2"), golden("B2", "P2")]))
    assert [i["id"] for i in load_golden(tmp_path, per_file=2)] == ["A1", "A2", "B1", "B2"]
    assert [i["id"] for i in load_golden(tmp_path, policy="P2")] == ["B1", "B2"]


def test_parse_stream_joins_tokens_and_keeps_only_contexts_sent_to_the_llm():
    lines = [json.dumps({"type": "token", "text": "Thirty "}), "", json.dumps({"type": "token", "text": "days."}),
             done([parent("clause A", True), parent("clause B", False), parent("clause C", True)])]
    r = parse_stream(lines)
    assert r["response"] == "Thirty days."
    assert r["contexts"] == ["clause A", "clause C"]
    assert (r["input_tokens"], r["output_tokens"], r["timings"]) == (100, 20, {"embed_ms": 300})


def test_parse_stream_rejects_a_stream_without_a_done_event():
    with pytest.raises(ValueError):
        parse_stream([json.dumps({"type": "token", "text": "partial"})])


def test_collect_passes_the_items_policy_and_records_a_failure_without_stopping():
    asked = []

    def ask(question, policy):
        asked.append((question, policy))
        if question == "q 2":
            raise ConnectionError("down")
        return parse_stream([json.dumps({"type": "token", "text": "ok"}), done([parent("c", True)])])

    records = list(collect([golden("1", "ReAssure 3.0"), golden("2", "HealthRecharge"), golden("3", "Health-Companion")], ask))
    assert asked == [("q 1", "ReAssure 3.0"), ("q 2", "HealthRecharge"), ("q 3", "Health-Companion")]
    assert [r.get("error") for r in records] == [None, "ConnectionError", None]
    assert records[0]["response"] == "ok" and records[0]["policy"] == "ReAssure 3.0"


def test_failed_and_unanswerable_items_are_not_scored():
    assert scorable({"question_type": "factual"})
    assert not scorable({"question_type": "unanswerable"})
    assert not scorable({"question_type": "factual", "error": "Timeout"})


def test_summary_reports_token_totals_and_metric_means_by_policy():
    records = [
        {**golden("1", "P1"), "response": "x", "input_tokens": 1000, "output_tokens": 100,
         "scores": {"faithfulness": 1.0, "context_recall": 0.5}},
        {**golden("2", "P1"), "response": "y", "input_tokens": 3000, "output_tokens": 300,
         "scores": {"faithfulness": 0.5, "context_recall": None}},
        {**golden("3", "P2", "unanswerable"), "response": "Not in the policy.", "input_tokens": 0, "output_tokens": 0},
    ]
    md = summarize(records, {"input_tokens": 50_000, "output_tokens": 7_000})
    assert "| System under test (answer generation) | 4,000 | 400 |" in md
    assert "| Judge (RAGAS metrics) | 50,000 | 7,000 |" in md
    assert "| Total | 54,000 | 7,400 |" in md
    assert "| P1 | 2 | 0.750 | 0.500 |" in md  # a failed (None) score is left out of the mean
    assert "**3** (P2)" in md
