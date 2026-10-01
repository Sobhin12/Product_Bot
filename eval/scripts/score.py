"""Stage 2: score collected answers with RAGAS, judged by a Databricks model (not the
model under test) through Databricks' OpenAI-compatible endpoint.

Uses ragas' `evaluate()` with the classic metric classes because that path reports the
judge's token usage (`token_usage_parser`); the newer `ragas.metrics.collections`
metrics do not. Pinned to ragas 0.4.3, where those classes are deprecated but present."""
import logging
import math
import os
import warnings
from collections import defaultdict

log = logging.getLogger("eval.score")

JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "databricks-gpt-oss-120b")
MAX_WORKERS = 4  # Databricks pay-per-token endpoints return 429 when pushed harder


def metrics():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (
            FactualCorrectness, Faithfulness, LLMContextPrecisionWithReference,
            LLMContextRecall, ResponseRelevancy,
        )
    return [LLMContextPrecisionWithReference(), LLMContextRecall(), Faithfulness(),
            FactualCorrectness(), ResponseRelevancy()]


def judge():
    """(llm, embeddings) for ragas: the judge LLM and the same GTE embeddings retrieval uses."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from retrieval import config

    base = f"{config.DATABRICKS_HOST}/serving-endpoints"
    llm = ChatOpenAI(model=JUDGE_MODEL, base_url=base, api_key=config.DATABRICKS_TOKEN,
                     temperature=0, max_retries=0, timeout=300)  # ragas retries (RunConfig)
    emb = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, base_url=base, api_key=config.DATABRICKS_TOKEN,
                           check_embedding_ctx_length=False)  # no tiktoken pre-split for a non-OpenAI model
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)


def scorable(record: dict) -> bool:
    """Unanswerable items have no reference context to judge against, and failed requests
    have no answer; both are reported, not scored."""
    return "error" not in record and record["question_type"] != "unanswerable"


def score(records: list[dict], llm, embeddings) -> tuple[list[dict], dict]:
    """Score the scorable records. Returns (records with a `scores` dict added, judge token
    totals). Metric failures become NaN scores (logged by ragas), not exceptions."""
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.cost import get_token_usage_for_openai
    from ragas.run_config import RunConfig

    todo = [r for r in records if scorable(r)]
    if not todo:
        return records, {"input_tokens": 0, "output_tokens": 0}
    dataset = EvaluationDataset([
        SingleTurnSample(user_input=r["question"], response=r["response"], retrieved_contexts=r["contexts"],
                         reference=r["answer"], reference_contexts=r["relevant_context"])
        for r in todo
    ])
    ms = metrics()
    log.info("score start items=%d metrics=%s judge=%s", len(todo), [m.name for m in ms], JUDGE_MODEL)
    result = evaluate(
        dataset, metrics=ms, llm=llm, embeddings=embeddings,
        run_config=RunConfig(max_workers=MAX_WORKERS, timeout=300, max_retries=6, max_wait=60),
        token_usage_parser=get_token_usage_for_openai, show_progress=False,
    )
    frame = result.to_pandas()
    for r, (_, row) in zip(todo, frame.iterrows()):
        r["scores"] = {m.name: _num(row[m.name]) for m in ms}
    usage = result.total_tokens()
    usage = usage if isinstance(usage, list) else [usage]  # one entry per judge model
    tokens = {"input_tokens": sum(u.input_tokens for u in usage), "output_tokens": sum(u.output_tokens for u in usage)}
    log.info("score done items=%d judge_input_tokens=%d judge_output_tokens=%d",
             len(todo), tokens["input_tokens"], tokens["output_tokens"])
    return records, tokens


def _num(v) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)


def summarize(records: list[dict], judge_tokens: dict) -> str:
    """Markdown report: mean of each metric overall / by policy / by question type, item
    counts, and token totals for the system under test and for the judge."""
    scored = [r for r in records if "scores" in r]
    names = list(scored[0]["scores"]) if scored else []

    def row(label: str, group: list[dict]) -> str:
        cells = []
        for m in names:
            vals = [r["scores"][m] for r in group if r["scores"][m] is not None]
            cells.append(f"{sum(vals) / len(vals):.3f}" if vals else "–")
        return f"| {label} | {len(group)} | " + " | ".join(cells) + " |"

    def table(title: str, key) -> list[str]:
        groups = defaultdict(list)
        for r in scored:
            groups[key(r)].append(r)
        head = [f"## {title}", "", "| | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        return head + [row(k, g) for k, g in sorted(groups.items())] + [""]

    gen_in = sum(r.get("input_tokens") or 0 for r in records)
    gen_out = sum(r.get("output_tokens") or 0 for r in records)
    errors = [r["id"] for r in records if "error" in r]
    unanswerable = [r for r in records if "error" not in r and r["question_type"] == "unanswerable"]
    lines = [
        "# RAGAS evaluation", "",
        f"Items: {len(records)} collected, {len(scored)} scored, {len(unanswerable)} unanswerable (not scored), "
        f"{len(errors)} failed requests{': ' + ', '.join(errors) if errors else ''}.", "",
        f"Judge: `{JUDGE_MODEL}`. Empty cells (–) mean every score for that metric failed.", "",
        "## Tokens", "",
        "| | input | output |", "|---|---|---|",
        f"| System under test (answer generation) | {gen_in:,} | {gen_out:,} |",
        f"| Judge (RAGAS metrics) | {judge_tokens['input_tokens']:,} | {judge_tokens['output_tokens']:,} |",
        f"| Total | {gen_in + judge_tokens['input_tokens']:,} | {gen_out + judge_tokens['output_tokens']:,} |", "",
        "Embedding tokens (query embedding, ResponseRelevancy) are not included.", "",
    ]
    if scored:
        lines += table("Overall", lambda r: "all") + table("By policy", lambda r: r["policy"]) \
            + table("By question type", lambda r: r["question_type"])
    if unanswerable:
        lines += ["## Unanswerable items (check by hand: the bot should say it is not in the policy)", ""]
        lines += [f"- **{r['id']}** ({r['policy']}) {r['question']}\n  - bot: {r['response'][:300]!r}" for r in unanswerable]
    return "\n".join(lines) + "\n"
