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
        answer:            The LLM's answer, or a fallback message if the
                           transcript evidence was too weak.
        top_k:             How many chunks were used as context.
        grounded:          True  → OpenAI was called with retrieved chunks.
                           False → similarity was below RAG_LOW_THRESHOLD;
                                   OpenAI was NOT called.
        confidence_level:  "high" → best chunk score ≥ RAG_GOOD_THRESHOLD.
                           "low"  → score between RAG_LOW_THRESHOLD and
                                    RAG_GOOD_THRESHOLD (OpenAI still called,
                                    but retrieval was borderline).
                           "none" → score below RAG_LOW_THRESHOLD; fallback
                                    returned without calling OpenAI.
        retrieved_chunks:  The chunks scored during retrieval (always present
                           so the caller can inspect scores).
    """

    question: str
    answer: str
    top_k: int
    grounded: bool
    confidence_level: str  # "high" | "low" | "none"
    retrieved_chunks: List[RetrievedChunkRead]
