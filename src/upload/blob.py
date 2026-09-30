"""Raw document storage: S3 in production, a local folder for development."""
import logging
import shutil
from pathlib import Path
from typing import Protocol

from . import config

log = logging.getLogger("upload")


class BlobStore(Protocol):
    def put(self, path: Path, key: str) -> None: ...

    def get(self, key: str, path: Path) -> None: ...

    def delete(self, key: str) -> None:
        """No error if the key does not exist."""


class S3BlobStore:
    def __init__(self, bucket: str):
        import boto3

        self.client = boto3.client("s3")
        self.bucket = bucket

    def put(self, path: Path, key: str) -> None:
        self.client.upload_file(str(path), self.bucket, key)

    def get(self, key: str, path: Path) -> None:
        self.client.download_file(self.bucket, key, str(path))

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)


class LocalBlobStore:
    def __init__(self, root: Path):
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key

    def put(self, path: Path, key: str) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)

    def get(self, key: str, path: Path) -> None:
        shutil.copyfile(self._path(key), path)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


def default_blob_store() -> BlobStore:
    if config.S3_DOCS_BUCKET:
        return S3BlobStore(config.S3_DOCS_BUCKET)
    log.warning("S3_DOCS_BUCKET is not set: raw uploads are stored in %s, not S3", config.LOCAL_BLOB_DIR)
    return LocalBlobStore(config.LOCAL_BLOB_DIR)
