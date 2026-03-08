from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TranscriptRead(BaseModel):
    """What the API returns when representing a transcript."""

    id: int
    video_id: int
    content: Optional[str]
    source: str
    created_at: datetime

    class Config:
        from_attributes = True
