import os

from dotenv import load_dotenv

from ingestion.config import OUTPUT_DIR, ROOT  # noqa: F401  (OUTPUT_DIR re-exported for the loader)

load_dotenv(ROOT / ".env")

DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "").strip().rstrip("/")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "").strip()
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "").strip()

S3_VECTORS_BUCKET = os.environ.get("S3_VECTORS_BUCKET", "").strip()
S3_VECTORS_INDEX = os.environ.get("S3_VECTORS_INDEX", "").strip()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://bot:bot@localhost:5433/bot")

EMBED_DIM = 1024
EMBED_MAX_TOKENS = 8192  # gte-large-en-v1.5 context
EMBED_BATCH = 8  # pay-per-token endpoint 429s at 16 inputs per call
# S3 Vectors allows up to 500 per PutVectors/DeleteVectors call, but boto3 sends each
# 1024-float vector as JSON text: ~22 KB at full float repr, ~12.7 KB once values are
# rounded to float32 precision (vectors.S3VectorStore.put). The endpoint closes a
# request whose body is still arriving after ~20s (measured: ConnectionClosedError at
# 22-25s), so a batch must upload well inside that even on a slow uplink. At the
# ~20 KB/s measured on a bad day, 20 vectors (~0.25 MB) take ~12s; 100 (~1.3 MB) cannot.
PUT_BATCH = 20
TOP_K = 10  # children per query, before parent dedup

# S3 Vectors client: explicit timeouts and our own retry, matching the LLM/embedding
# clients rather than relying on boto3's defaults.
S3_CONNECT_TIMEOUT = 10
S3_READ_TIMEOUT = 60
S3_MAX_ATTEMPTS = 5
S3_BACKOFF_BASE = 1.0
S3_BACKOFF_CAP = 20.0
