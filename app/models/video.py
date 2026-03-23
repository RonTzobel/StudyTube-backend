from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class Video(SQLModel, table=True):
    """
    Represents a video uploaded by a user.

    At this stage we only store basic metadata.
    Future additions: file storage path/URL, duration, processing status,
    embeddings reference, language, etc.
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
    # This is the canonical reference to the video file.
    # The bucket name and region come from settings — never stored here.
    s3_key: Optional[str] = Field(default=None)

    # Local filesystem path — kept for backward compatibility with videos
    # uploaded before S3 migration, and as a temp path during worker processing.
    file_path: Optional[str] = Field(default=None)

    # Processing states: uploaded → queued → processing → transcribed → indexing → ready | failed
    status: str = Field(default="uploaded")

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
