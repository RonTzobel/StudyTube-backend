"""
Tutor router — AI study assistant endpoints.

These endpoints expose the tutor_service layer over REST.
They follow the same structure as app/routers/videos.py:
  - thin HTTP layer only
  - all logic delegated to the service
  - consistent error handling (ValueError → 4xx, RuntimeError → 500)

Endpoints
---------
  POST /tutor/videos/{video_id}/summary   — structured AI study summary
  POST /tutor/videos/{video_id}/explain   — explain a concept via RAG
  POST /tutor/videos/{video_id}/quiz      — generate a multiple-choice quiz
  GET  /tutor/discover?topic=...          — find relevant lectures by topic
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session

from app.database.session import get_session
from app.schemas.tutor import (
    ExplainRequest,
    ExplainResponse,
    RelevantVideoResult,
    StudySummaryResponse,
    StudyQuizResponse,
    TutorQuizRequest,
)
from app.services.tutor_service import (
    explain_concept_from_video,
    find_relevant_videos_for_topic,
    generate_video_study_quiz,
    summarize_video_for_study,
)

router = APIRouter(prefix="/tutor", tags=["Tutor"])


@router.post("/videos/{video_id}/summary", response_model=StudySummaryResponse)
def tutor_summarize(
    video_id: int,
    session: Session = Depends(get_session),
):
    """
    Generate an AI-powered study summary for a video's transcript.

    Returns a structured summary with:
      - A paragraph summary of the lecture
      - Bullet-point key takeaways
      - Important terms with brief definitions

    Prerequisites:
      - POST /api/v1/videos/{video_id}/transcribe must have been called.
      - OPENAI_API_KEY must be set in .env.
    """
    try:
        return summarize_video_for_study(session, video_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/videos/{video_id}/explain", response_model=ExplainResponse)
def tutor_explain(
    video_id: int,
    request: ExplainRequest,
    session: Session = Depends(get_session),
):
    """
    Explain a concept from a video using RAG.

    Internally uses the same retrieval + OpenAI pipeline as /api/v1/videos/ask.
    The tutor layer adds a stable response shape and a consistent `mode` field.

    Request body:
      { "question": "Explain JWT authentication", "top_k": 3 }

    Prerequisites:
      - POST /api/v1/videos/{video_id}/transcribe, /chunk, and /embed must have been called.
      - OPENAI_API_KEY must be set in .env.
    """
    try:
        return explain_concept_from_video(
            session, video_id, request.question, request.top_k
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/videos/{video_id}/quiz", response_model=StudyQuizResponse)
def tutor_quiz(
    video_id: int,
    request: TutorQuizRequest,
    session: Session = Depends(get_session),
):
    """
    Generate a multiple-choice study quiz for a video.

    Samples transcript chunks evenly across the whole video so questions
    cover the full lecture, not just one section. All questions are grounded
    in the transcript — the LLM is instructed not to invent facts.

    Request body:
      { "num_questions": 5 }

    Prerequisites:
      - POST /api/v1/videos/{video_id}/transcribe, /chunk, and /embed must have been called.
      - OPENAI_API_KEY must be set in .env.
    """
    try:
        return generate_video_study_quiz(session, video_id, request.num_questions)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/discover", response_model=List[RelevantVideoResult])
def tutor_discover(
    topic: str = Query(..., min_length=1, description="Topic or concept to search for across all lectures."),
    session: Session = Depends(get_session),
):
    """
    Find which lectures are most relevant to a topic.

    Runs semantic search across all embedded videos and returns them
    ranked by relevance score (highest first). Videos that have not been
    through the full pipeline (transcribe → chunk → embed) are silently skipped.

    Query parameter:
      ?topic=JWT authentication

    Returns an empty list if no videos are embedded or none are relevant.
    """
    # user_id=1: auth placeholder until JWT is implemented.
    return find_relevant_videos_for_topic(session, user_id=1, topic=topic)
