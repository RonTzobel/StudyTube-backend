"""
Schemas for the /tutor REST endpoints.

These are separate from the existing video/qa/quiz schemas so the tutor
API can evolve its response shapes independently. All tutor responses
include a `mode` field so clients know which tutor action produced them.
"""

from typing import List

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ExplainRequest(BaseModel):
    """Request body for POST /tutor/videos/{video_id}/explain."""
    question: str = Field(..., min_length=1, description="Concept or question to explain from the video.")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of transcript chunks to use as context.")


class TutorQuizRequest(BaseModel):
    """Request body for POST /tutor/videos/{video_id}/quiz."""
    num_questions: int = Field(default=5, ge=1, le=20, description="Number of quiz questions to generate.")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class StudySummaryResponse(BaseModel):
    """Response from POST /tutor/videos/{video_id}/summary."""
    video_id: int
    video_title: str
    mode: str                    # always "study_summary"
    summary: str
    key_points: List[str]
    important_terms: List[str]


class ExplainResponse(BaseModel):
    """Response from POST /tutor/videos/{video_id}/explain."""
    video_id: int
    video_title: str
    mode: str                    # always "explain"
    question: str
    answer: str
    confidence_level: str        # "high" | "low" | "none"


class StudyQuizQuestion(BaseModel):
    """A single multiple-choice question inside StudyQuizResponse."""
    question: str
    options: List[str]           # always 4 options: ["A. ...", "B. ...", ...]
    correct_answer: str


class StudyQuizResponse(BaseModel):
    """Response from POST /tutor/videos/{video_id}/quiz."""
    video_id: int
    video_title: str
    mode: str                    # always "quiz"
    num_questions: int
    questions: List[StudyQuizQuestion]


class RelevantVideoResult(BaseModel):
    """A single result item from GET /tutor/discover."""
    video_id: int
    title: str
    best_similarity: float
    top_excerpt: str
