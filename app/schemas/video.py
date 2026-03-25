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

    status values:
      "uploaded"     → file saved, pipeline not yet started
      "queued"       → job accepted by Redis, worker not yet picked it up
      "processing"   → worker extracting audio with ffmpeg
      "transcribing" → worker running Whisper transcription
      "embedding"    → chunking + embedding in progress
      "completed"    → all done; quiz, search, and ask are available
      "failed"       → pipeline error; see error_message for details
    """
    id: int
    title: str
    description: Optional[str]
    original_filename: Optional[str]
    s3_key: Optional[str]
    status: str
    error_message: Optional[str]
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

    202 — pipeline dispatched to background (status: "queued").
    200 — video already fully processed (status: "completed").

    Poll GET /api/v1/videos/{video_id} and watch `status`:
      "queued"       → job in Redis, waiting for a free worker
      "processing"   → audio extraction running
      "transcribing" → Whisper transcription running
      "embedding"    → chunking + embedding in progress
      "completed"    → all done; quiz, search, and ask are available
      "failed"       → pipeline error; see error_message field for details
    """
    message: str
    video_id: int
    status: str
