from datetime import datetime

from pydantic import BaseModel


class SummaryRead(BaseModel):
    """What the API returns when representing a summary."""

    id: int
    video_id: int
    content: str
    source: str
    created_at: datetime

    class Config:
        from_attributes = True
