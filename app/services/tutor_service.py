"""
tutor_service.py — Application-layer tutor orchestration.

This is a thin wrapper around the existing lower-level services.
It exists so both the REST API (app/routers/tutor.py) and the MCP server
(app/mcp_server.py) share the same tutor logic without duplicating it.

All heavy logic (retrieval, QA, quiz generation) stays in the lower-level
services — this file only orchestrates calls and shapes the output into
stable tutor-oriented payloads.

Functions
---------
  summarize_video_for_study      → OpenAI structured summary
  explain_concept_from_video     → qa_service.answer_question()
  generate_video_study_quiz      → quiz_service.generate_quiz()
  find_relevant_videos_for_topic → retrieval_service.search_chunks() per video
"""

import json

from sqlmodel import Session

from app.config.settings import settings
from app.services.qa_service import answer_question
from app.services.quiz_service import generate_quiz
from app.services.retrieval_service import search_chunks
from app.services.transcript_service import get_transcript_by_video_id
from app.services.video_service import get_video_by_id, get_videos_for_user


# ---------------------------------------------------------------------------
# OpenAI client — same lazy-import pattern used across the codebase.
# ---------------------------------------------------------------------------

def _get_openai_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "The 'openai' package is not installed. Run: pip install openai"
        )
    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )
    return OpenAI(api_key=settings.OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# 1. summarize_video_for_study
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM = (
    "You are a study assistant that helps students learn from lecture transcripts. "
    "Read the provided transcript and produce a structured study summary. "
    "Return ONLY valid JSON with this exact format — no markdown, no extra text:\n"
    '{"summary": "...", "key_points": ["...", "..."], "important_terms": ["term: definition", ...]}'
)


def summarize_video_for_study(session: Session, video_id: int) -> dict:
    """
    Produce a structured AI study summary for a video's transcript.

    Uses OpenAI to generate a paragraph summary, a bullet list of key points,
    and a list of important terms with brief definitions.

    Args:
        session:   DB session.
        video_id:  The video to summarize.

    Returns:
        dict with: video_id, video_title, mode, summary, key_points, important_terms

    Raises:
        ValueError:   If the video or transcript does not exist.
        RuntimeError: If OpenAI is not installed or OPENAI_API_KEY is missing.
    """
    video = get_video_by_id(session, video_id)
    if video is None:
        raise ValueError(f"Video {video_id} not found.")

    transcript = get_transcript_by_video_id(session, video_id)
    if transcript is None or not transcript.content:
        raise ValueError(
            f"No transcript found for video {video_id}. "
            "Run transcription first."
        )

    client = _get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {"role": "user",   "content": f"Transcript:\n{transcript.content}"},
        ],
        temperature=0.3,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Graceful fallback: treat the raw text as the summary.
        parsed = {"summary": raw, "key_points": [], "important_terms": []}

    return {
        "video_id": video_id,
        "video_title": video.title,
        "mode": "study_summary",
        "summary": parsed.get("summary", ""),
        "key_points": parsed.get("key_points", []),
        "important_terms": parsed.get("important_terms", []),
    }


# ---------------------------------------------------------------------------
# 2. explain_concept_from_video
# ---------------------------------------------------------------------------

def explain_concept_from_video(
    session: Session,
    video_id: int,
    question: str,
    top_k: int = 3,
) -> dict:
    """
    Explain a concept from a video using the existing RAG pipeline.

    Delegates entirely to qa_service.answer_question() — no second QA pipeline.

    Args:
        session:   DB session.
        video_id:  The video to query.
        question:  The concept or question to explain.
        top_k:     Number of chunks to retrieve as context (default 3).

    Returns:
        dict with: video_id, video_title, mode, question, answer, confidence_level

    Raises:
        ValueError:   If the video does not exist or has no embedded chunks.
        RuntimeError: If OpenAI is not installed or OPENAI_API_KEY is missing.
    """
    video = get_video_by_id(session, video_id)
    if video is None:
        raise ValueError(f"Video {video_id} not found.")

    result = answer_question(session, video_id, question, top_k)

    return {
        "video_id": video_id,
        "video_title": video.title,
        "mode": "explain",
        "question": result.question,
        "answer": result.answer,
        "confidence_level": result.confidence_level,
    }


# ---------------------------------------------------------------------------
# 3. generate_video_study_quiz
# ---------------------------------------------------------------------------

def generate_video_study_quiz(
    session: Session,
    video_id: int,
    num_questions: int = 5,
) -> dict:
    """
    Generate a multiple-choice study quiz grounded in a video's transcript.

    Delegates entirely to quiz_service.generate_quiz(). Chunks are sampled
    evenly across the whole transcript for broad coverage.

    Args:
        session:       DB session.
        video_id:      The video to quiz on.
        num_questions: How many questions to generate (default 5).

    Returns:
        dict with: video_id, video_title, mode, num_questions, questions

    Raises:
        ValueError:   If the video has no embedded chunks, or OpenAI returns bad JSON.
        RuntimeError: If OpenAI is not installed or OPENAI_API_KEY is missing.
    """
    video = get_video_by_id(session, video_id)
    if video is None:
        raise ValueError(f"Video {video_id} not found.")

    quiz = generate_quiz(session, video_id, num_questions=num_questions)

    return {
        "video_id": video_id,
        "video_title": video.title,
        "mode": "quiz",
        "num_questions": quiz.num_questions,
        "questions": [
            {
                "question": q.question,
                "options": q.options,
                "correct_answer": q.correct_answer,
            }
            for q in quiz.questions
        ],
    }


# ---------------------------------------------------------------------------
# 4. find_relevant_videos_for_topic
# ---------------------------------------------------------------------------

def find_relevant_videos_for_topic(
    session: Session,
    user_id: int,
    topic: str,
) -> list[dict]:
    """
    Search across a user's videos to find which lectures cover a topic.

    Runs a one-chunk semantic search against every video that has been embedded,
    then returns results ranked by best similarity score (highest first).
    Videos with no embedded chunks are silently skipped.

    Args:
        session:  DB session.
        user_id:  The user whose videos to search.
        topic:    A topic, concept, or question to search for.

    Returns:
        List of dicts, each with: video_id, title, best_similarity, top_excerpt.
        Empty list if no videos have been embedded or none are relevant.
    """
    videos = get_videos_for_user(session, user_id)
    results = []

    for video in videos:
        try:
            chunks = search_chunks(session, video.id, topic, top_k=1)
            if chunks:
                results.append({
                    "video_id": video.id,
                    "title": video.title,
                    "best_similarity": round(chunks[0].similarity_score, 3),
                    "top_excerpt": chunks[0].content[:200],
                })
        except ValueError:
            # Video has no embedded chunks yet — skip silently.
            pass

    results.sort(key=lambda x: x["best_similarity"], reverse=True)
    return results
