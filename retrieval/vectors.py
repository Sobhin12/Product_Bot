"""Vector store adapters: Amazon S3 Vectors for real use, in-memory for tests."""
import math
from dataclasses import dataclass
from typing import Protocol

from . import config


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

        self.client = boto3.client("s3vectors")
        self.bucket = bucket or config.S3_VECTORS_BUCKET
        self.index = index or config.S3_VECTORS_INDEX

    def put(self, items: list[dict]) -> None:
        for batch in _batches(items, config.PUT_BATCH):
            self.client.put_vectors(
                vectorBucketName=self.bucket,
                indexName=self.index,
                vectors=[
                    {"key": i["key"], "data": {"float32": i["vector"]}, "metadata": i["metadata"]}
                    for i in batch
                ],
            )

    def query(self, vector: list[float], doc_ids: list[str], top_k: int) -> list[Hit]:
        r = self.client.query_vectors(
            vectorBucketName=self.bucket,
            indexName=self.index,
            queryVector={"float32": vector},
            topK=top_k,
            filter={"doc_id": {"$in": doc_ids}},
            returnMetadata=True,
            returnDistance=True,
        )
        return [Hit(v["key"], v["metadata"]["parent_id"], v["distance"]) for v in r["vectors"]]

    def delete(self, keys: list[str]) -> None:
        for batch in _batches(keys, config.PUT_BATCH):
            self.client.delete_vectors(vectorBucketName=self.bucket, indexName=self.index, keys=batch)


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
