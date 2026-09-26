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
PUT_BATCH = 500  # S3 Vectors PutVectors/DeleteVectors per-call maximum
TOP_K = 10  # children per query, before parent dedup
