from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class TranscriptChunk(SQLModel, table=True):
    """
    One piece of a transcript, ready for embedding.

    A transcript is split into overlapping fixed-size chunks so that a
    vector search can retrieve the most relevant passage for a given query.
    Each chunk stores its position in the original text (start_char /
    end_char) so it can be mapped back to the full transcript if needed.

    Embedding storage:
        embedding   — the vector stored as a JSON string (list of floats).
                      Using JSON avoids a pgvector dependency for now.
                      When you switch to pgvector, replace this column type;
                      the rest of the code stays the same.
        is_embedded — False until embed() has run on this chunk. Use this
                      flag to find un-embedded chunks without scanning content.

    Relationships:
        video_id → videos.id  (the video this chunk came from)
    """

    __tablename__ = "transcript_chunks"

    id: Optional[int] = Field(default=None, primary_key=True)

    # The video this chunk belongs to (no FK to transcripts — video_id is enough)
    video_id: int = Field(foreign_key="videos.id", nullable=False, index=True)

    # Position of this chunk in the ordered sequence (0-based)
    chunk_index: int = Field(nullable=False)

    # The actual text of this chunk
    content: str = Field(nullable=False)

    # Character offsets in the original transcript — useful for highlighting
    start_char: Optional[int] = Field(default=None)
    end_char: Optional[int] = Field(default=None)

    # Embedding vector stored as a JSON string, e.g. "[0.12, -0.34, ...]"
    # None means this chunk has not been embedded yet.
    embedding: Optional[str] = Field(default=None)

    # True once embedding has been generated and saved for this chunk
    is_embedded: bool = Field(default=False)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
