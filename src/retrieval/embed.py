"""Databricks embedding endpoint. gte-large-en takes no query/document prefix."""
import logging
import time

import requests

from . import config

log = logging.getLogger("retrieval")

RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 6

# One pooled session for every call (query and ingestion threads alike): a new TCP+TLS
# connection to the workspace costs ~1s from India, more than the embedding itself.
_session = requests.Session()


def _post(texts: list[str]) -> list[list[float]]:
    url = f"{config.DATABRICKS_HOST}/serving-endpoints/{config.EMBEDDING_MODEL}/invocations"
    headers = {"Authorization": f"Bearer {config.DATABRICKS_TOKEN}"}
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = _session.post(url, headers=headers, json={"input": texts}, timeout=60)
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            reason = type(e).__name__
        else:
            if r.status_code not in RETRY_STATUS:
                r.raise_for_status()
                data = sorted(r.json()["data"], key=lambda d: d["index"])
                return [d["embedding"] for d in data]
            if attempt == MAX_ATTEMPTS - 1:
                r.raise_for_status()
            reason = f"HTTP {r.status_code}"
        delay = 3 * 2**attempt  # workspace-wide QPS limit: back off generously
        log.warning("embedding retry attempt=%d/%d error=%s delay=%.1fs", attempt + 1, MAX_ATTEMPTS, reason, delay)
        time.sleep(delay)
    raise AssertionError("unreachable")


def embed_query(text: str) -> list[float]:
    return _post([text])[0]


def embed_documents(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), config.EMBED_BATCH):
        out += _post(texts[i : i + config.EMBED_BATCH])
    return out
