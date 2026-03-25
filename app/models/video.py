from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class Video(SQLModel, table=True):
    """
    Represents a video uploaded by a user.

    Processing status lifecycle:
        uploaded    → file accepted, pipeline not started
        queued      → job sent to Redis, worker hasn't picked it up yet
        processing  → worker is extracting audio with ffmpeg
        transcribing → worker is running Whisper transcription
        embedding   → worker is chunking + generating embeddings
        completed   → fully processed; search, quiz, and ask are available
        failed      → pipeline error; see error_message for details
    """

    __tablename__ = "videos"

    id: Optional[int] = Field(default=None, primary_key=True)

    # The user who uploaded this video
    user_id: int = Field(foreign_key="users.id", nullable=False, index=True)

    title: str = Field(nullable=False)
    description: Optional[str] = Field(default=None)

    # Original filename as provided by the uploader (display only).
    original_filename: Optional[str] = Field(default=None)

    # S3 object key, e.g. "videos/7/abc123.mp4".
    # The bucket name and region come from settings — never stored here.
    s3_key: Optional[str] = Field(default=None)

    # Local filesystem path — kept for backward compatibility with videos
    # uploaded before S3 migration, and as a temp path during worker processing.
    file_path: Optional[str] = Field(default=None)

    # Current pipeline status — see docstring above for valid values.
    status: str = Field(default="uploaded")

    # Human-readable description of the last pipeline failure.
    # Populated by the worker on exception; None on success or before first run.
    error_message: Optional[str] = Field(default=None)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
