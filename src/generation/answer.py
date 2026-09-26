"""Generation block: query -> embed -> retrieve -> prompt -> LLM, streamed.

    async for chunk in generator.answer("What is the cataract waiting period?", "ReAssure"):
        ...

Embedding and retrieval run outside the semaphore; only the LLM call (including the
whole stream) holds a slot, so LLM concurrency is bounded across all users.
"""
import asyncio
import logging
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Iterator

from retrieval.retrieve import Parent

from . import config
from .llm import LLMClient
from .prompt import build_prompt

log = logging.getLogger("generation")

_DONE = object()


class Generator:
    def __init__(
        self,
        llm: LLMClient,
        retrieve: Callable[[str, str], list[Parent]],
        concurrency: int = config.LLM_CONCURRENCY,
    ):
        self.llm = llm
        self.retrieve = retrieve  # (query, policy) -> parents, blocking
        self.semaphore = asyncio.Semaphore(concurrency)

    async def answer(self, query: str, policy: str) -> AsyncIterator[str]:
        """Yield the answer as text chunks. Never raises: any failure becomes the generic
        fallback message for the user, with the real error logged under a trace_id."""
        trace_id = uuid.uuid4().hex[:12]
        try:
            parents = await asyncio.to_thread(self.retrieve, query, policy)
        except Exception:
            log.exception("retrieval failed trace_id=%s", trace_id)
            yield config.FALLBACK_MESSAGE
            return
        if not parents:
            yield config.NO_CONTEXT_MESSAGE
            return

        system, user = build_prompt(query, parents)
        started = False
        try:
            async with self.semaphore:
                async for chunk in self._stream(system, user):
                    started = True
                    yield chunk
        except Exception:
            log.exception("llm call failed trace_id=%s", trace_id)
            yield ("\n\n" if started else "") + config.FALLBACK_MESSAGE

    async def _stream(self, system: str, user: str) -> AsyncIterator[str]:
        """Bridge the blocking token iterator into async, one thread per call."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def put(item) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def worker() -> None:
            try:
                for token in self.llm.stream(system, user):
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

    def run(query: str, policy: str) -> list[Parent]:
        with db.connect() as conn:  # one short-lived connection per query
            return retrieve(conn, store, embed_query, query, policy)

    llm = BedrockLLM() if config.LLM_PROVIDER == "bedrock" else DatabricksLLM()
    return Generator(llm, run)
