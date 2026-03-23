from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class VideoCreate(BaseModel):
    """Data required to register a new video."""
    title: str
    description: Optional[str] = None


class VideoRead(BaseModel):
    """
    What the API returns when representing a video.

    Intentionally omits:
      - user_id   : internal FK, the client already knows who they are
      - file_path : server-side path, must never be exposed to clients
    """
    id: int
    title: str
    description: Optional[str]
    original_filename: Optional[str]
    s3_key: Optional[str]
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    """Returned by POST /upload after a successful S3 upload."""
    message: str
    video_id: int
    s3_key: str
    status: str


class VideoUpdate(BaseModel):
    """Fields the client can update. All optional."""
    title: Optional[str] = None
    description: Optional[str] = None


class TranscribeAccepted(BaseModel):
    """
    Returned by POST /transcribe.

    202 — pipeline dispatched to background (status: "processing").
    200 — video already ready, nothing started (status: "ready").

    Poll GET /api/v1/videos/{video_id} and check `status`:
      "processing"  → audio extraction + Whisper running
      "transcribed" → transcript done, chunking about to start
      "indexing"    → chunking / embedding in progress
      "ready"       → all done; quiz, chat, and search are available
      "failed"      → pipeline error (check server logs)
    """
    message: str
    video_id: int
    status: str  # "processing" | "ready"
