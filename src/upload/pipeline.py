"""In-process ingestion: one asyncio task per document, at most INGESTION_CONCURRENCY at a time.

Stages: S3 write -> scan -> parse -> chunk -> embed/store -> indexed. A failure anywhere
marks the document failed; a retry re-runs everything from the start."""
import asyncio
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import psycopg
import pymupdf

from ingestion import config as ingestion_config
from ingestion.run import process
from retrieval import db
from retrieval.load import load_document
from retrieval.vectors import VectorStore

from . import config, lifecycle, repo
from .blob import BlobStore

log = logging.getLogger("upload")


class Cancelled(Exception):
    """The document was deleted while it was being ingested."""


class ScanError(Exception):
    pass


def scan_pdf(path: Path) -> None:
    try:
        doc = pymupdf.open(path)
    except Exception as e:
        raise ScanError("not a readable PDF") from e
    with doc:
        if doc.needs_pass:
            raise ScanError("PDF is password protected")
        if doc.page_count == 0:
            raise ScanError("PDF has no pages")
        if doc.page_count > config.MAX_PAGES:
            raise ScanError(f"PDF has more than {config.MAX_PAGES} pages")


class IngestionRunner:
    def __init__(
        self,
        blobs: BlobStore,
        store: VectorStore,
        embed: Callable[[list[str]], list[list[float]]],
        parse: Callable = process,
        connect: Callable[[], psycopg.Connection] = lambda: db.connect(autocommit=True),
        concurrency: int = config.INGESTION_CONCURRENCY,
    ):
        self.blobs, self.store, self.embed, self.parse, self.connect = blobs, store, embed, parse, connect
        self.semaphore = asyncio.Semaphore(concurrency)
        self.running: set[str] = set()  # queued or in progress in this process
        self._tasks: set[asyncio.Task] = set()

    def schedule(self, doc_id: str) -> None:
        """Start ingesting `doc_id` in the background, subject to the concurrency limit."""
        self.running.add(doc_id)
        task = asyncio.create_task(self._run(doc_id))
        self._tasks.add(task)  # a task with no reference can be garbage-collected mid-run
        task.add_done_callback(self._tasks.discard)

    async def wait_idle(self) -> None:
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def _run(self, doc_id: str) -> None:
        try:
            async with self.semaphore:
                await asyncio.to_thread(self._ingest, doc_id)
        except Exception:
            log.exception("ingestion crashed doc_id=%s", doc_id)
        finally:
            self.running.discard(doc_id)

    def _ingest(self, doc_id: str) -> None:
        """Runs in a worker thread. Never raises: failures are recorded on the document."""
        stage = "starting"
        with self.connect() as conn:
            try:

                def move_to(new_stage: str) -> None:
                    nonlocal stage
                    stage = new_stage
                    if not repo.set_status(conn, doc_id, new_stage):
                        raise Cancelled

                doc = repo.get(conn, doc_id)
                if doc is None or doc["status"] == "deleted":
                    return
                path = lifecycle.local_path(doc_id)

                stage = "storing"
                if path.exists():
                    self.blobs.put(path, doc["s3_key"])
                else:  # retrying after a restart: the working copy is gone, S3 has the file
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self.blobs.get(doc["s3_key"], path)

                move_to("scanning")
                scan_pdf(path)
                parents, children = self.parse(path, on_stage=move_to)  # sets parsing, then chunking
                move_to("embedding")
                load_document(conn, self.store, self.embed, doc_id, parents, children)
                move_to("indexed")
                self._finish(conn, doc)
                path.unlink(missing_ok=True)
                shutil.rmtree(ingestion_config.OUTPUT_DIR / path.stem, ignore_errors=True)
            except Cancelled:
                log.info("ingestion cancelled, document deleted doc_id=%s", doc_id)
            except Exception as e:
                # Detail goes to the log only; the stored message is safe to show a user.
                log.exception("ingestion failed doc_id=%s stage=%s", doc_id, stage)
                detail = str(e) if isinstance(e, ScanError) else type(e).__name__
                repo.mark_failed(conn, doc_id, f"Failed during {stage}: {detail}")

    def _finish(self, conn: psycopg.Connection, doc: dict) -> None:
        """A replacement has been indexed, so the document it replaces can go."""
        old = doc["replaces_doc_id"]
        if not old:
            return
        try:
            lifecycle.remove_document(conn, str(old), self.store, self.blobs)
        except lifecycle.LifecycleError:
            pass  # already gone
        except Exception:
            # New content is live; the old copy stays (duplicate content, not missing content).
            log.exception("could not delete replaced document doc_id=%s replaced=%s", doc["doc_id"], old)
            return
        repo.clear_replaces(conn, str(doc["doc_id"]))  # the name is now this document's alone

    def _revive_stuck(self) -> list[str]:
        """Blocking part of the sweep: reset documents stranded in a non-terminal status
        (e.g. by a restart) to pending, and return their ids."""
        revived: list[str] = []
        with self.connect() as conn:
            for doc in repo.stuck(conn, config.STUCK_AFTER_SECONDS):
                doc_id = str(doc["doc_id"])
                if doc_id in self.running:
                    continue  # slow, not stuck
                repo.mark_failed(conn, doc_id, f"Interrupted during {doc['status']}")
                if repo.start_retry(conn, doc_id):
                    revived.append(doc_id)
                else:
                    log.error("stuck document out of retries doc_id=%s", doc_id)
        return revived

    async def sweep_once(self) -> list[str]:
        revived = await asyncio.to_thread(self._revive_stuck)
        for doc_id in revived:
            self.schedule(doc_id)  # on the event loop, not in the worker thread
        return revived

    async def sweep_forever(self) -> None:
        while True:
            await asyncio.sleep(config.SWEEP_INTERVAL_SECONDS)
            try:
                if revived := await self.sweep_once():
                    log.warning("sweep restarted %d stuck document(s)", len(revived))
            except Exception:
                log.exception("sweep failed")
