"""RAGAS evaluation of the running Policy Bot API against the golden dataset.

    python eval/scripts/run.py --per-file 2                # smoke: first 2 items of each golden file
    python eval/scripts/run.py --policy "ReAssure 3.0"     # one policy, all its items
    python eval/scripts/run.py --rescore eval/results/<run> # score saved answers again, no API calls

Needs the API up (POLICY_BOT_API_URL, default http://localhost:8765) for collection, and
DATABRICKS_HOST / DATABRICKS_TOKEN / EMBEDDING_MODEL in .env for the judge. Writes
eval/results/<timestamp>/{responses.jsonl, scores.jsonl, summary.md}."""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from collect import collect, http_ask, load_golden
from score import judge, score, summarize

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
log = logging.getLogger("eval")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", help="only items of this policy")
    p.add_argument("--per-file", type=int, help="only the first N items of each golden file")
    p.add_argument("--api", default=os.environ.get("POLICY_BOT_API_URL", "http://localhost:8765"))
    p.add_argument("--rescore", type=Path, help="run dir whose responses.jsonl to score again")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "openai", "urllib3", "ragas"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    start = time.perf_counter()
    if args.rescore:
        run_dir = args.rescore
        records = [json.loads(line) for line in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
        log.info("rescore run_dir=%s items=%d", run_dir, len(records))
    else:
        items = load_golden(policy=args.policy, per_file=args.per_file)
        if not items:
            log.error("no golden items match policy=%r", args.policy)
            return 1
        try:
            requests.get(f"{args.api}/health", timeout=5).raise_for_status()
        except requests.RequestException as e:
            log.error("API not reachable at %s (%s); start it first", args.api, type(e).__name__)
            return 1
        run_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True)
        log.info("collect start items=%d api=%s run_dir=%s", len(items), args.api, run_dir)
        records = []
        with (run_dir / "responses.jsonl").open("w", encoding="utf-8") as out:
            for record in collect(items, http_ask(args.api)):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()  # keep what was collected if the run is interrupted
                records.append(record)
        log.info(
            "collect done items=%d failed=%d input_tokens=%d output_tokens=%d",
            len(records), sum("error" in r for r in records),
            sum(r.get("input_tokens") or 0 for r in records), sum(r.get("output_tokens") or 0 for r in records),
        )

    llm, embeddings = judge()
    records, judge_tokens = score(records, llm, embeddings)
    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = summarize(records, judge_tokens)
    (run_dir / "summary.md").write_text(report, encoding="utf-8")

    gen_in = sum(r.get("input_tokens") or 0 for r in records)
    gen_out = sum(r.get("output_tokens") or 0 for r in records)
    log.info(
        "eval done run_dir=%s generation_tokens=%d/%d judge_tokens=%d/%d total_tokens=%d/%d (input/output) wall_s=%.0f",
        run_dir, gen_in, gen_out, judge_tokens["input_tokens"], judge_tokens["output_tokens"],
        gen_in + judge_tokens["input_tokens"], gen_out + judge_tokens["output_tokens"], time.perf_counter() - start,
    )
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
