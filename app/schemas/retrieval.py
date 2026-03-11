from typing import Optional

from pydantic import BaseModel, Field


class SearchChunksRequest(BaseModel):
    """
    Request body for the semantic chunk search endpoint.

    Fields:
        query:   The natural-language question or phrase to search for.
        top_k:   How many of the most relevant chunks to return.
                 Defaults to 5; capped at 20 to keep responses manageable.
    """

    query: str = Field(..., min_length=1, description="The search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")


class RetrievedChunkRead(BaseModel):
    """
    One search result: a transcript chunk ranked by semantic similarity.

    Fields:
        id:               Primary key of the TranscriptChunk row.
        chunk_index:      Position of this chunk in the transcript (0-based).
        content:          The raw text of the chunk.
        similarity_score: Cosine similarity between the query and this chunk
                          (float between -1.0 and 1.0; higher = more relevant).
        start_char:       Starting character offset in the original transcript.
        end_char:         Ending character offset in the original transcript.
    """

    id: int
    chunk_index: int
    content: str
    similarity_score: float
    start_char: Optional[int]
    end_char: Optional[int]

    class Config:
        from_attributes = True
