"""FastAPI app.

    uvicorn api.main:app --port 8000

No authentication or rate limiting yet (HLD: reused from the hosting site's session).
"""
import asyncio
import contextlib
import logging
from dataclasses import dataclass

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from generation.answer import Generator
from retrieval import db
from retrieval.vectors import VectorStore
from upload import lifecycle, repo
from upload.blob import BlobStore
from upload.pipeline import IngestionRunner

log = logging.getLogger("api")


@dataclass
class Services:
    runner: IngestionRunner
    generator: Generator
    store: VectorStore
    blobs: BlobStore


def default_services() -> Services:
    from generation.answer import build_default
    from retrieval.embed import embed_documents
    from retrieval.vectors import S3VectorStore
    from upload.blob import default_blob_store

    store, blobs = S3VectorStore(), default_blob_store()
    return Services(IngestionRunner(blobs, store, embed_documents), build_default(), store, blobs)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    policy: str = Field(min_length=1, max_length=100)


def _run_with_conn(fn, *args):
    with db.connect(autocommit=True) as conn:
        return fn(conn, *args)


async def _db(fn, *args):
    """Run a blocking lifecycle/repo function on a worker thread with its own connection."""
    try:
        return await asyncio.to_thread(_run_with_conn, fn, *args)
    except lifecycle.LifecycleError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


def _public(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k not in ("content_hash", "s3_key")}


def create_app(services: Services | None = None) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or await asyncio.to_thread(default_services)
        await _db(lambda conn: db.init_schema(conn))
        sweeper = asyncio.create_task(app.state.services.runner.sweep_forever())
        yield
        sweeper.cancel()

    app = FastAPI(title="Policy Bot", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/upload")
    async def upload(
        file: UploadFile = File(...),
        display_name: str = Form(...),
        is_update: bool = Form(False),
        replaces_doc_id: str | None = Form(None),
    ):
        result = await _db(
            lifecycle.upload, file.file, file.filename or "", display_name, is_update, replaces_doc_id or None
        )
        if result["status"] == "accepted":
            app.state.services.runner.schedule(result["doc_id"])
            return JSONResponse(result, status_code=202)
        return result

    @app.get("/documents")
    async def list_documents(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
        return [_public(d) for d in await _db(repo.list_live, limit, offset)]

    @app.get("/documents/{doc_id}")
    async def get_document(doc_id: str):
        return _public(await _db(lifecycle.get_document, doc_id))

    @app.post("/documents/{doc_id}/retry", status_code=202)
    async def retry(doc_id: str):
        await _db(lifecycle.start_retry, doc_id)
        app.state.services.runner.schedule(doc_id)
        return {"doc_id": doc_id, "status": "pending"}

    @app.delete("/documents/{doc_id}", status_code=204)
    async def delete(doc_id: str):
        s = app.state.services
        await _db(lifecycle.delete_document, doc_id, s.store, s.blobs)

    @app.post("/query")
    async def query(req: QueryRequest):
        """Streams the answer as plain text. Failures arrive as the generic fallback message."""
        return StreamingResponse(
            app.state.services.generator.answer(req.query, req.policy),
            media_type="text/plain; charset=utf-8",
        )

    return app


app = create_app()
