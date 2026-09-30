"""Generation block: query -> embed -> retrieve -> prompt -> LLM, streamed.

    async for event in generator.events("What is the cataract waiting period?", "ReAssure"):
        ...  # {"type": "token", "text": ...} pieces, then one {"type": "done", ...}

    async for chunk in generator.answer(...):
        ...  # just the answer text, for callers that don't need token/timing/parent data

Embedding and retrieval run outside the semaphore; only the LLM call (including the
whole stream) holds a slot, so LLM concurrency is bounded across all users.
"""
import asyncio
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator

from retrieval.retrieve import Parent

from . import config
from .llm import LLMClient
from .prompt import build_prompt

log = logging.getLogger("generation")

_DONE = object()


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


class Generator:
    def __init__(
        self,
        llm: LLMClient,
        retrieve: Callable[[str, str, dict], list[Parent]],
        concurrency: int = config.LLM_CONCURRENCY,
    ):
        self.llm = llm
        self.retrieve = retrieve  # (query, policy, timings) -> parents, blocking; fills timings in place
        self.semaphore = asyncio.Semaphore(concurrency)

    async def events(self, query: str, policy: str) -> AsyncIterator[dict]:
        """Yield {"type": "token", "text": ...} pieces of the answer, followed by exactly
        one final {"type": "done", "input_tokens", "output_tokens", "latency_ms", "timings",
        "parents"} event. `latency_ms` covers this whole call, start to finish. `timings`
        holds per-stage milliseconds for the stages that ran: whatever retrieve reports
        (embed_ms, search_ms, db_ms), then retrieval_ms (all of retrieval, connection
        included), llm_wait_ms (queued for an LLM slot), ttft_ms (slot acquired to first
        token, retries included) and generation_ms (first token to end of stream). Token counts are
        None whenever the LLM was never reached or the call failed - a partial count
        from a failed stream is not reported as if it were the real total. `parents` is
        every parent retrieval returned, each with "kept": True if it was actually sent
        to the model or False if the context budget left it out. Never raises: a failure
        becomes the generic fallback message, logged under a trace_id."""
        trace_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        usage: dict = {}
        timings: dict = {}  # written by the retrieval thread, read only after it returns
        parents_out: list[dict] = []
        stage_start = time.perf_counter()
        try:
            parents = await asyncio.to_thread(self.retrieve, query, policy, timings)
        except Exception:
            log.exception("retrieval failed trace_id=%s", trace_id)
            parents = None
        timings["retrieval_ms"] = _ms(stage_start)
        if parents is None:
            yield {"type": "token", "text": config.FALLBACK_MESSAGE}
            yield self._done(trace_id, start, usage, timings, parents_out)
            return
        if not parents:
            yield {"type": "token", "text": config.NO_CONTEXT_MESSAGE}
            yield self._done(trace_id, start, usage, timings, parents_out)
            return

        prompt = build_prompt(query, parents)
        parents_out = [self._parent_info(p, kept=True) for p in prompt.kept] + [
            self._parent_info(p, kept=False) for p in prompt.dropped
        ]
        started = False
        stage_start = time.perf_counter()
        try:
            async with self.semaphore:
                timings["llm_wait_ms"] = _ms(stage_start)
                stage_start = time.perf_counter()
                async for chunk in self._stream(prompt.system, prompt.user, usage):
                    if not started:
                        timings["ttft_ms"] = _ms(stage_start)
                    started = True
                    yield {"type": "token", "text": chunk}
                timings["generation_ms"] = _ms(stage_start) - timings.get("ttft_ms", 0)
        except Exception:
            log.exception("llm call failed trace_id=%s", trace_id)
            usage.clear()  # a partial count is misleading once the call has failed
            yield {"type": "token", "text": ("\n\n" if started else "") + config.FALLBACK_MESSAGE}
        yield self._done(trace_id, start, usage, timings, parents_out)

    async def answer(self, query: str, policy: str) -> AsyncIterator[str]:
        """Just the answer text - see `events` for token/timing/parent data."""
        async for event in self.events(query, policy):
            if event["type"] == "token":
                yield event["text"]

    @staticmethod
    def _parent_info(p: Parent, kept: bool) -> dict:
        return {
            "parent_id": p.parent_id,
            "title": p.title,
            "section": p.section,
            "clauses": p.clauses,
            "source_pages": p.source_pages,
            "text": p.text,
            "kept": kept,
        }

    @staticmethod
    def _done(trace_id: str, start: float, usage: dict, timings: dict, parents: list[dict]) -> dict:
        """The final event, also logged as one line per query (no query or answer text)."""
        done = {
            "type": "done",
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "latency_ms": _ms(start),
            "timings": timings,
            "parents": parents,
        }
        log.info(
            "query trace_id=%s latency_ms=%d input_tokens=%s output_tokens=%s parents=%d kept=%d timings=%s",
            trace_id, done["latency_ms"], done["input_tokens"], done["output_tokens"],
            len(parents), sum(p["kept"] for p in parents), timings,
        )
        return done

    async def _stream(self, system: str, user: str, usage: dict) -> AsyncIterator[str]:
        """Bridge the blocking token iterator into async, one thread per call. `usage` is
        written by that thread; safe to read once this generator returns (the thread has
        by then finished - the _DONE sentinel is only sent after the loop below exits)."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def put(item) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def worker() -> None:
            try:
                for token in self.llm.stream(system, user, usage):
                    if stop.is_set():
                        break
                    put(token)
            except BaseException as e:
                put(e)
            finally:
                put(_DONE)

        loop.run_in_executor(None, worker)
        try:
            while (item := await queue.get()) is not _DONE:
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            # A caller that stops early (client disconnect) ends the worker at its next token.
            stop.set()


def build_default() -> Generator:
    """Wire the real Postgres, Databricks embeddings, S3 Vectors and the configured LLM."""
    from retrieval import db
    from retrieval.embed import embed_query
    from retrieval.retrieve import retrieve
    from retrieval.vectors import S3VectorStore

    from .llm import BedrockLLM, DatabricksLLM

    store = S3VectorStore()

    def run(query: str, policy: str, timings: dict) -> list[Parent]:
        with db.connect() as conn:  # one short-lived connection per query
            return retrieve(conn, store, embed_query, query, policy, timings=timings)

    llm = BedrockLLM() if config.LLM_PROVIDER == "bedrock" else DatabricksLLM()
    return Generator(llm, run)
