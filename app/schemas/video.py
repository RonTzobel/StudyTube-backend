from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class VideoCreate(BaseModel):
    """Data required to register a new video."""

    title: str
    description: Optional[str] = None


class VideoRead(BaseModel):
    """What the API returns when representing a video."""

    id: int
    user_id: int
    title: str
    description: Optional[str]
    file_path: Optional[str]
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class VideoUpdate(BaseModel):
    """Fields the client can update. All are optional."""

    title: Optional[str] = None
    description: Optional[str] = None
