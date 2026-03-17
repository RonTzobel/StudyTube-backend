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
      - user_id   : internal FK, the authenticated client already knows who they are
      - file_path : server-side filesystem path, must never be exposed to clients
    """
    id: int
    title: str
    description: Optional[str]
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class VideoUpdate(BaseModel):
    """Fields the client can update. All optional."""
    title: Optional[str] = None
    description: Optional[str] = None
