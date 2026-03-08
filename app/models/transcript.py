from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class Transcript(SQLModel, table=True):
    """
    Stores the text transcript of a video.

    One video has one transcript. The transcript is the raw text
    that will later be chunked and embedded for RAG queries.

    Future additions: language, chunk table (for embeddings), speaker labels.
    """

    __tablename__ = "transcripts"

    id: Optional[int] = Field(default=None, primary_key=True)

    # Each transcript belongs to exactly one video
    video_id: int = Field(foreign_key="videos.id", unique=True, nullable=False)

    # The full raw transcript text
    content: Optional[str] = Field(default=None)

    # Source of the transcript: "whisper", "upload", "youtube", etc.
    source: str = Field(default="pending")

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
