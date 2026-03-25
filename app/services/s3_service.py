"""
s3_service.py — AWS S3 helpers for video file storage.

Why S3 keys instead of full URLs in the database?
    The bucket name and region live in settings and can change (staging →
    production, region migration). Storing only the key keeps the DB row
    stable — you can always reconstruct a URL or presigned link from the
    key + settings at any time, without a data migration.

Why upload_fileobj instead of put_object?
    upload_fileobj uses the S3 multipart upload API automatically for large
    files. It streams the content in chunks so a 500 MB video file is never
    fully loaded into memory at once.

Why is upload_fileobj considered successful when no exception is raised?
    boto3's upload_fileobj raises botocore.exceptions.ClientError on any
    S3-side failure (auth error, bucket missing, network timeout, etc.).
    If the call returns normally, S3 has acknowledged receipt of all parts
    and the object is durably stored. There is no separate "check status"
    step needed.
"""

import logging
import os
import tempfile
import uuid
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from app.config.settings import settings

_log = logging.getLogger(__name__)


def _s3():
    """Return a boto3 S3 client configured from environment settings."""
    return boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
    )


def upload_file_to_s3(file: "UploadFile", user_id: int) -> str:
    """
    Stream-upload a video file to S3 and return the S3 key.

    Key format: videos/{user_id}/{uuid}{ext}
    Example:    videos/7/3f2a1b8e...c9d0.mp4

    Args:
        file:     The UploadFile from the FastAPI request (already seeked to 0).
        user_id:  Owner's user ID — used as the first path segment in the key.

    Returns:
        The S3 object key (e.g. "videos/7/abc123.mp4").

    Raises:
        ClientError: If the upload fails for any reason.
    """
    suffix = Path(file.filename or "video.bin").suffix or ".bin"
    key = f"videos/{user_id}/{uuid.uuid4().hex}{suffix}"

    _log.info("s3 upload start | key=%s  bucket=%s", key, settings.AWS_S3_BUCKET)
    try:
        _s3().upload_fileobj(
            file.file,
            settings.AWS_S3_BUCKET,
            key,
            ExtraArgs={
                "ContentType": file.content_type or "application/octet-stream"
            },
        )
    except ClientError as exc:
        _log.error("s3 upload failed | key=%s | %s", key, exc)
        raise

    _log.info("s3 upload done | key=%s", key)
    return key


def delete_file_from_s3(s3_key: str) -> None:
    """
    Delete an object from S3.

    Silently ignores NoSuchKey — if the file is already gone the desired
    outcome (object does not exist) is already achieved.

    Raises:
        ClientError: For any S3 error other than NoSuchKey.
    """
    _log.info("s3 delete | key=%s  bucket=%s", s3_key, settings.AWS_S3_BUCKET)
    try:
        _s3().delete_object(Bucket=settings.AWS_S3_BUCKET, Key=s3_key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchKey":
            _log.error("s3 delete failed | key=%s | %s", s3_key, exc)
            raise


def download_file_from_s3(s3_key: str) -> str:
    """
    Download an S3 object to a local temp file and return its path.

    Used by the worker before running ffmpeg — ffmpeg requires a local path.
    The CALLER must delete the temp file when done (use try/finally).

    Returns:
        Absolute path to the temporary local file.

    Raises:
        ClientError: If the download fails.
    """
    suffix = Path(s3_key).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()

    _log.info("s3 download start | key=%s  dest=%s", s3_key, tmp.name)
    try:
        _s3().download_file(settings.AWS_S3_BUCKET, s3_key, tmp.name)
    except ClientError as exc:
        os.unlink(tmp.name)
        _log.error("s3 download failed | key=%s | %s", s3_key, exc)
        raise

    size = Path(tmp.name).stat().st_size
    _log.info("s3 download done | key=%s  bytes=%d", s3_key, size)
    return tmp.name
