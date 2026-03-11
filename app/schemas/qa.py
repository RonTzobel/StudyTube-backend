from typing import List

from pydantic import BaseModel, Field

from app.schemas.retrieval import RetrievedChunkRead


class AskRequest(BaseModel):
    """
    Request body for the RAG question-answering endpoint.

    Fields:
        question:  The natural-language question about the video content.
        top_k:     How many transcript chunks to retrieve as context.
                   More chunks = more context for the LLM, but also more
                   tokens and slightly slower responses. 3–5 is a good default.
    """

    question: str = Field(..., min_length=1, description="Question about the video")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of chunks to use as context")


class AskResponse(BaseModel):
    """
    Response from the RAG question-answering endpoint.

    Fields:
        question:          The original question echoed back.
        answer:            The LLM's answer, grounded in the retrieved chunks.
        top_k:             How many chunks were used as context.
        retrieved_chunks:  The chunks that were passed to the LLM — useful for
                           debugging or showing the user where the answer came from.
    """

    question: str
    answer: str
    top_k: int
    retrieved_chunks: List[RetrievedChunkRead]
