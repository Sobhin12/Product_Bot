import os

from ingestion.config import ROOT
from retrieval import config as _retrieval  # noqa: F401  (loads .env)

# Raw uploads are kept in this bucket. Unset -> a local folder is used (dev only).
S3_DOCS_BUCKET = os.environ.get("S3_DOCS_BUCKET", "").strip()

UPLOAD_DIR = ROOT / "data" / "uploads"  # working copies while a document is ingested
LOCAL_BLOB_DIR = ROOT / "data" / "blobs"

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_PAGES = 500
READ_CHUNK = 1024 * 1024
MAX_NAME_LENGTH = 200

INGESTION_CONCURRENCY = 2  # pipelines at once; bounded by the embedding endpoint's QPS limit and Docling's CPU use
MAX_RETRIES = 3

STUCK_AFTER_SECONDS = 600  # a non-terminal document untouched this long is presumed lost in a restart
SWEEP_INTERVAL_SECONDS = 60

ACTIVE_STATUSES = ("pending", "scanning", "parsing", "chunking", "embedding")
