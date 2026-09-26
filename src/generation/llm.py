"""Streaming LLM clients (Databricks, Bedrock). Retries with backoff only until the first token is out:
after that, retrying would duplicate text the caller has already sent to the user."""
import json
import random
import time
from collections.abc import Callable, Iterator
from typing import Protocol

import requests

from retrieval import config as retrieval_config

from . import config

RETRYABLE_HTTP = {429, 500, 502, 503, 504}
RETRYABLE_CODES = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "ModelNotReadyException",
    "InternalServerException",
}


class LLMClient(Protocol):
    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield answer text as it is generated. Raises on failure."""


def is_retryable(exc: BaseException) -> bool:
    from botocore.exceptions import ClientError, ConnectionError, ReadTimeoutError

    if isinstance(exc, requests.HTTPError):
        return exc.response is not None and exc.response.status_code in RETRYABLE_HTTP
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True  # ChunkedEncodingError (dropped stream) is a ConnectionError
    if isinstance(exc, ClientError):
        return exc.response.get("Error", {}).get("Code") in RETRYABLE_CODES
    return isinstance(exc, (ConnectionError, ReadTimeoutError))


def stream_with_retry(
    open_stream: Callable[[], Iterator[str]],
    *,
    retryable: Callable[[BaseException], bool] = is_retryable,
    max_attempts: int = config.LLM_MAX_ATTEMPTS,
    base: float = config.LLM_BACKOFF_BASE,
    cap: float = config.LLM_BACKOFF_CAP,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random = random.Random(),
) -> Iterator[str]:
    """Yield from `open_stream()`. A retryable failure before the first token opens a
    new stream after a full-jitter delay, uniform in [0, min(cap, base * 2**attempt)].
    A failure after the first token, a non-retryable one, or the last attempt raises."""
    for attempt in range(max_attempts):
        started = False
        try:
            for token in open_stream():
                started = True
                yield token
            return
        except Exception as e:
            if started or attempt == max_attempts - 1 or not retryable(e):
                raise
            sleep(rng.uniform(0, min(cap, base * 2**attempt)))


class BedrockLLM:
    def __init__(self, model_id: str | None = None):
        import boto3
        from botocore.config import Config

        self.model_id = model_id or config.BEDROCK_MODEL_ID
        self.client = boto3.client(
            "bedrock-runtime",
            # One attempt: retries are ours, so they are counted and jittered in one place.
            config=Config(
                retries={"max_attempts": 1, "mode": "standard"},
                connect_timeout=config.LLM_CONNECT_TIMEOUT,
                read_timeout=config.LLM_READ_TIMEOUT,
            ),
        )

    def _open(self, system: str, user: str) -> Iterator[str]:
        response = self.client.converse_stream(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": config.LLM_MAX_TOKENS, "temperature": config.LLM_TEMPERATURE},
        )
        for event in response["stream"]:
            delta = event.get("contentBlockDelta", {}).get("delta", {})
            if "text" in delta:
                yield delta["text"]

    def stream(self, system: str, user: str) -> Iterator[str]:
        return stream_with_retry(lambda: self._open(system, user))


class DatabricksLLM:
    """Databricks Foundation Model API chat endpoint (OpenAI-compatible, SSE streaming)."""

    def __init__(self, model: str | None = None):
        self.model = model or config.DATABRICKS_LLM_MODEL

    def _open(self, system: str, user: str) -> Iterator[str]:
        url = f"{retrieval_config.DATABRICKS_HOST}/serving-endpoints/{self.model}/invocations"
        body = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE,
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {retrieval_config.DATABRICKS_TOKEN}"}
        with requests.post(
            url, headers=headers, json=body, stream=True,
            timeout=(config.LLM_CONNECT_TIMEOUT, config.LLM_READ_TIMEOUT),
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                choices = json.loads(data).get("choices") or []
                text = choices[0].get("delta", {}).get("content") if choices else None
                if text:
                    yield text

    def stream(self, system: str, user: str) -> Iterator[str]:
        return stream_with_retry(lambda: self._open(system, user))
