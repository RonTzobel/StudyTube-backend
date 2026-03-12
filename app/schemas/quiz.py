from typing import List

from pydantic import BaseModel, Field


class QuizRequest(BaseModel):
    """
    Request body for the quiz-generation endpoint.

    Fields:
        num_questions:  How many multiple-choice questions to generate.
                        Capped at 20 to keep prompt size and cost reasonable.
        top_k:          How many transcript chunks to pass to OpenAI as context.
                        Chunks are sampled evenly across the transcript so that
                        questions cover the whole video, not just one section.
    """

    num_questions: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of multiple-choice questions to generate",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Number of transcript chunks to use as context (sampled evenly)",
    )


class QuizQuestion(BaseModel):
    """
    A single multiple-choice question.

    Fields:
        question:       The question text.
        options:        Exactly 4 answer choices, e.g. ["A. ...", "B. ...", ...].
        correct_answer: The full text of the correct option, e.g. "A. recursion".
    """

    question: str
    options: List[str]      # always 4 items
    correct_answer: str


class QuizResponse(BaseModel):
    """
    Response from the quiz-generation endpoint.

    Fields:
        video_id:       The video this quiz was generated for.
        num_questions:  How many questions were actually generated
                        (may be fewer than requested if the transcript is short).
        questions:      The list of multiple-choice questions.
    """

    video_id: int
    num_questions: int
    questions: List[QuizQuestion]
