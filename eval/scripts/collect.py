"""Stage 1: ask the running API every golden question, with the item's policy as the
metadata filter, and keep what came back (answer, the contexts actually sent to the
LLM, token counts, timings). Saved so scoring can be re-run without asking again."""
import json
import logging
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import requests

log = logging.getLogger("eval.collect")

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden"
TIMEOUT = 300  # seconds; covers the whole streamed answer


def load_golden(golden_dir: Path = GOLDEN_DIR, policy: str | None = None, per_file: int | None = None) -> list[dict]:
    """Every golden item, optionally only one policy and/or the first `per_file` of each file."""
    items = []
    for f in sorted(golden_dir.glob("*.json")):
        rows = [r for r in json.loads(f.read_text(encoding="utf-8")) if policy in (None, r["policy"])]
        items += rows[:per_file] if per_file else rows
    return items


def parse_stream(lines: Iterable[str]) -> dict:
    """Fold the NDJSON stream of /query into one record. `contexts` are the texts of the
    parents that were really sent to the LLM (kept), in rank order."""
    text, done = [], None
    for line in lines:
        if not line.strip():
            continue
        event = json.loads(line)
        if event["type"] == "token":
            text.append(event["text"])
        elif event["type"] == "done":
            done = event
    if done is None:
        raise ValueError("stream ended without a done event")
    kept = [p for p in done["parents"] if p["kept"]]
    return {
        "response": "".join(text),
        "contexts": [p["text"] for p in kept],
        "context_pages": [p["source_pages"] for p in kept],
        "input_tokens": done["input_tokens"],
        "output_tokens": done["output_tokens"],
        "latency_ms": done["latency_ms"],
        "timings": done["timings"],
    }


def http_ask(api_url: str) -> Callable[[str, str], dict]:
    session = requests.Session()

    def ask(question: str, policy: str) -> dict:
        with session.post(f"{api_url}/query", json={"query": question, "policy": policy},
                          stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            return parse_stream(r.iter_lines(decode_unicode=True))

    return ask


def collect(items: list[dict], ask: Callable[[str, str], dict]) -> Iterator[dict]:
    """One record per golden item. A failed request is recorded with `error`, not raised,
    so one bad item does not lose the rest of the run."""
    for n, item in enumerate(items, 1):
        record = {k: item[k] for k in ("id", "policy", "question", "answer", "relevant_context", "question_type")}
        start = time.perf_counter()
        try:
            record.update(ask(item["question"], item["policy"]))
        except Exception as e:
            log.exception("collect failed id=%s policy=%s", item["id"], item["policy"])
            record["error"] = type(e).__name__
        else:
            log.info(
                "collect %d/%d id=%s policy=%r contexts=%d input_tokens=%s output_tokens=%s latency_ms=%s wall_ms=%d",
                n, len(items), item["id"], item["policy"], len(record["contexts"]),
                record["input_tokens"], record["output_tokens"], record["latency_ms"],
                (time.perf_counter() - start) * 1000,
            )
        yield record
