"""Vector store adapters: Amazon S3 Vectors for real use, in-memory for tests."""
import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Protocol

from . import config

log = logging.getLogger("retrieval")

RETRYABLE_CLIENT_CODES = {"ThrottlingException", "InternalServerException", "ServiceUnavailableException"}


def _retryable(exc: BaseException) -> bool:
    """Connection dropped/timed out, or AWS's own throttling/5xx - never a bad request.
    botocore splits transport-layer failures across two base classes: ConnectionError
    (SSLError, ConnectTimeoutError, EndpointConnectionError, ProxyConnectionError) and
    HTTPClientError (ConnectionClosedError, ReadTimeoutError, ResponseStreamingError).
    Retry on both bases rather than enumerating subclasses one at a time - but not on
    the much broader BotoCoreError, which also covers things like a bad parameter or
    missing credentials that retrying would never fix."""
    from botocore.exceptions import ClientError, ConnectionError, HTTPClientError

    if isinstance(exc, ClientError):
        return exc.response.get("Error", {}).get("Code") in RETRYABLE_CLIENT_CODES
    return isinstance(exc, (ConnectionError, HTTPClientError))


def call_with_retry(call, *, max_attempts: int = config.S3_MAX_ATTEMPTS):
    """Run `call()`, retrying a transient failure with full-jitter exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return call()
        except Exception as e:
            if attempt == max_attempts - 1 or not _retryable(e):
                raise
            delay = random.uniform(0, min(config.S3_BACKOFF_CAP, config.S3_BACKOFF_BASE * 2**attempt))
            response = getattr(e, "response", None)  # ClientError: report AWS's error code
            reason = response.get("Error", {}).get("Code") if isinstance(response, dict) else type(e).__name__
            log.warning("s3vectors retry attempt=%d/%d error=%s delay=%.1fs", attempt + 1, max_attempts, reason, delay)
            time.sleep(delay)


@dataclass
class Hit:
    chunk_key: str
    parent_id: str
    distance: float


class VectorStore(Protocol):
    def put(self, items: list[dict]) -> None:
        """items: {"key", "vector", "metadata": {doc_id, parent_id, chunk_key}}"""

    def query(self, vector: list[float], doc_ids: list[str], top_k: int) -> list[Hit]:
        """Nearest first, restricted to the given doc_ids."""

    def delete(self, keys: list[str]) -> None: ...


def _batches(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


class S3VectorStore:
    def __init__(self, bucket: str | None = None, index: str | None = None):
        import boto3
        from botocore.config import Config

        self.client = boto3.client(
            "s3vectors",
            # One attempt: retries are ours (call_with_retry), so they are counted and
            # jittered in one place, same rationale as the Bedrock/Databricks clients.
            config=Config(
                retries={"max_attempts": 1, "mode": "standard"},
                connect_timeout=config.S3_CONNECT_TIMEOUT,
                read_timeout=config.S3_READ_TIMEOUT,
            ),
        )
        self.bucket = bucket or config.S3_VECTORS_BUCKET
        self.index = index or config.S3_VECTORS_INDEX

    def put(self, items: list[dict]) -> None:
        for batch in _batches(items, config.PUT_BATCH):
            vectors = [{"key": i["key"], "data": {"float32": i["vector"]}, "metadata": i["metadata"]} for i in batch]
            call_with_retry(
                lambda vectors=vectors: self.client.put_vectors(
                    vectorBucketName=self.bucket, indexName=self.index, vectors=vectors
                )
            )

    def query(self, vector: list[float], doc_ids: list[str], top_k: int) -> list[Hit]:
        r = call_with_retry(
            lambda: self.client.query_vectors(
                vectorBucketName=self.bucket,
                indexName=self.index,
                queryVector={"float32": vector},
                topK=top_k,
                filter={"doc_id": {"$in": doc_ids}},
                returnMetadata=True,
                returnDistance=True,
            )
        )
        return [Hit(v["key"], v["metadata"]["parent_id"], v["distance"]) for v in r["vectors"]]

    def delete(self, keys: list[str]) -> None:
        for batch in _batches(keys, config.PUT_BATCH):
            call_with_retry(
                lambda batch=batch: self.client.delete_vectors(
                    vectorBucketName=self.bucket, indexName=self.index, keys=batch
                )
            )


class InMemoryVectorStore:
    def __init__(self):
        self.items: dict[str, dict] = {}

    def put(self, items: list[dict]) -> None:
        for i in items:
            self.items[i["key"]] = i

    def query(self, vector: list[float], doc_ids: list[str], top_k: int) -> list[Hit]:
        def dist(v: list[float]) -> float:
            dot = sum(a * b for a, b in zip(vector, v))
            norm = math.sqrt(sum(a * a for a in vector)) * math.sqrt(sum(b * b for b in v))
            return 1 - dot / norm

        hits = [
            Hit(i["key"], i["metadata"]["parent_id"], dist(i["vector"]))
            for i in self.items.values()
            if i["metadata"]["doc_id"] in doc_ids
        ]
        return sorted(hits, key=lambda h: h.distance)[:top_k]

    def delete(self, keys: list[str]) -> None:
        for k in keys:
            self.items.pop(k, None)
