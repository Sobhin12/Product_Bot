"""Streaming LLM clients (Databricks, Bedrock). Retries with backoff only until the first token is out:
after that, retrying would duplicate text the caller has already sent to the user."""
import json
import logging
import random
import time
from collections.abc import Callable, Iterator
from typing import Protocol

import requests

from retrieval import config as retrieval_config

from . import config

log = logging.getLogger("generation")

RETRYABLE_HTTP = {429, 500, 502, 503, 504}
RETRYABLE_CODES = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "ModelNotReadyException",
    "InternalServerException",
}


class LLMClient(Protocol):
    def stream(self, system: str, user: str, usage: dict) -> Iterator[str]:
        """Yield answer text as it is generated. Raises on failure. On success, mutates
        `usage` in place with {"input_tokens", "output_tokens"} once the provider reports
        them (a fresh dict per call - callers must not share one across concurrent calls).
        Left empty if the provider never reports them or the call fails first."""


def is_retryable(exc: BaseException) -> bool:
    from botocore.exceptions import ClientError, ConnectionError, ReadTimeoutError

    if isinstance(exc, requests.HTTPError):
        return exc.response is not None and exc.response.status_code in RETRYABLE_HTTP
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True  # ChunkedEncodingError (dropped stream) is a ConnectionError
    if isinstance(exc, ClientError):
        return exc.response.get("Error", {}).get("Code") in RETRYABLE_CODES
    return isinstance(exc, (ConnectionError, ReadTimeoutError))


def describe(exc: BaseException) -> str:
    """Short, secret-free label for a retry log line: HTTP status, AWS error code, or type."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return f"HTTP {exc.response.status_code}"
    response = getattr(exc, "response", None)
    if isinstance(response, dict) and (code := response.get("Error", {}).get("Code")):
        return code
    return type(exc).__name__


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
            delay = rng.uniform(0, min(cap, base * 2**attempt))
            log.warning("llm retry attempt=%d/%d error=%s delay=%.1fs", attempt + 1, max_attempts, describe(e), delay)
            sleep(delay)


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

    def _open(self, system: str, user: str, usage: dict) -> Iterator[str]:
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
            if usage_event := event.get("metadata", {}).get("usage"):
                usage["input_tokens"] = usage_event.get("inputTokens")
                usage["output_tokens"] = usage_event.get("outputTokens")

    def stream(self, system: str, user: str, usage: dict) -> Iterator[str]:
        return stream_with_retry(lambda: self._open(system, user, usage))


class DatabricksLLM:
    """Databricks Foundation Model API chat endpoint (OpenAI-compatible, SSE streaming)."""

    def __init__(self, model: str | None = None):
        self.model = model or config.DATABRICKS_LLM_MODEL
        # Pooled keep-alive connections: a fresh TCP+TLS handshake costs ~1s from India.
        self.session = requests.Session()

    def _open(self, system: str, user: str, usage: dict) -> Iterator[str]:
        url = f"{retrieval_config.DATABRICKS_HOST}/serving-endpoints/{self.model}/invocations"
        body = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE,
            "stream": True,
            "stream_options": {"include_usage": True},  # each chunk carries running totals; the last one wins
        }
        headers = {"Authorization": f"Bearer {retrieval_config.DATABRICKS_TOKEN}"}
        with self.session.post(
            url, headers=headers, json=body, stream=True,
            timeout=(config.LLM_CONNECT_TIMEOUT, config.LLM_READ_TIMEOUT),
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    # Read on to the end of the body rather than returning: a response
                    # closed part-read drops its connection instead of pooling it.
                    continue
                obj = json.loads(data)
                if usage_obj := obj.get("usage"):
                    usage["input_tokens"] = usage_obj.get("prompt_tokens")
                    usage["output_tokens"] = usage_obj.get("completion_tokens")
                choices = obj.get("choices") or []
                text = choices[0].get("delta", {}).get("content") if choices else None
                if text:
                    yield text

    def stream(self, system: str, user: str, usage: dict) -> Iterator[str]:
        return stream_with_retry(lambda: self._open(system, user, usage))
