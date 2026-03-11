import json
from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, field_validator


class ChunkRead(BaseModel):
    """What the API returns when representing a single transcript chunk."""

    id: int
    video_id: int
    chunk_index: int
    content: str
    start_char: Optional[int]
    end_char: Optional[int]
    # None when not yet embedded; list of floats once embedding is generated.
    # The DB stores this as a JSON string — the validator below parses it.
    embedding: Optional[List[float]]
    is_embedded: bool
    created_at: datetime

    @field_validator("embedding", mode="before")
    @classmethod
    def parse_embedding(cls, v: Any) -> Optional[List[float]]:
        """Accept either a JSON string (from DB) or an already-parsed list."""
        if v is None:
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v

    class Config:
        from_attributes = True
