from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class Summary(SQLModel, table=True):
    """
    Stores a generated summary for a video's transcript.

    One video has at most one summary (enforced by unique=True on video_id).
    The summary is generated from the transcript content — either locally
    or via an external AI service (e.g. OpenAI) in the future.
    """

    __tablename__ = "summaries"

    id: Optional[int] = Field(default=None, primary_key=True)

    # Each summary belongs to exactly one video
    video_id: int = Field(foreign_key="videos.id", unique=True, nullable=False)

    # The generated summary text
    content: str = Field(nullable=False)

    # How the summary was produced: "local" now, "openai" later
    source: str = Field(default="local")

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
