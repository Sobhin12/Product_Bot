import os

from retrieval import config as _retrieval  # noqa: F401  (loads .env)

# "databricks" or "bedrock"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "databricks").strip().lower()
DATABRICKS_LLM_MODEL = os.environ.get("DATABRICKS_LLM_MODEL", "databricks-meta-llama-3-3-70b-instruct").strip()

# Amazon Bedrock model / cross-region inference profile used for answers.
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0").strip()

LLM_CONCURRENCY = 4  # global cap on in-flight LLM calls, across all users
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.0

# Timeouts (seconds) on the LLM HTTP connection; no reliance on SDK defaults.
LLM_CONNECT_TIMEOUT = 10
LLM_READ_TIMEOUT = 60

# Retry before the first token only: exponential backoff with full jitter.
LLM_MAX_ATTEMPTS = 4
LLM_BACKOFF_BASE = 1.0
LLM_BACKOFF_CAP = 20.0

# Retrieved-context budget in tokens (ingestion.split.count_tokens estimate).
CONTEXT_BUDGET_TOKENS = 6000

FALLBACK_MESSAGE = "Something went wrong. Please try again in a moment."
NO_CONTEXT_MESSAGE = "I couldn't find relevant information in the selected policy documents."
