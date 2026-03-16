"""
StudyTube MCP Server — Phases 1-3
-----------------------------------
A thin stdio adapter on top of the StudyTube service layer.
Does NOT replace FastAPI — run both independently.

Tools
-----
  list_videos          → video_service.get_videos_for_user()
  get_transcript       → Transcript table (direct query)
  ask_video            → qa_service.answer_question()             [RAG QA]
  summarize_video      → tutor_service.summarize_video_for_study()
  quiz_video           → tutor_service.generate_video_study_quiz()
  find_relevant_video  → tutor_service.find_relevant_videos_for_topic()

Dev / inspect
-------------
  PYTHONPATH=. mcp dev app/mcp_server.py

Run via stdio (Claude Desktop / any MCP client)
-----------------------------------------------
  PYTHONPATH=. python -m app.mcp_server

Claude Desktop (claude_desktop_config.json)
-------------------------------------------
  {
    "mcpServers": {
      "studytube": {
        "command": "C:/Users/Ronzo/StudyTube/backend/venv311/Scripts/python.exe",
        "args": ["-m", "app.mcp_server"],
        "cwd": "C:/Users/Ronzo/StudyTube/backend",
        "env": { "PYTHONPATH": "C:/Users/Ronzo/StudyTube/backend" }
      }
    }
  }
"""

import json
import logging
import sys

from sqlmodel import Session, select

from mcp.server.fastmcp import FastMCP

from app.database.session import engine
from app.models.transcript import Transcript
from app.services.qa_service import answer_question
from app.services.tutor_service import (
    find_relevant_videos_for_topic,
    generate_video_study_quiz,
    summarize_video_for_study,
)
from app.services.video_service import get_video_by_id, get_videos_for_user

# MCP stdio uses stdout exclusively for JSON-RPC communication.
# Route ALL logging to stderr and raise the threshold to WARNING so that
# SQLAlchemy's SQL echo (INFO level) and library chatter never reach stdout.
# force=True removes any stdout handlers that imported libraries may have added.
logging.basicConfig(level=logging.WARNING, stream=sys.stderr, force=True)

# Belt-and-suspenders: disable SQLAlchemy query echo directly on the engine.
# session.py sets echo=settings.DEBUG which is True in .env — fine for FastAPI,
# but would corrupt the stdio stream here.
engine.echo = False

logger = logging.getLogger(__name__)

mcp = FastMCP("studytube")


# ---------------------------------------------------------------------------
# Tools — thin wrappers around the service layer.
# FastMCP builds the JSON schema from type hints; docstrings become the
# tool description shown to the LLM.
# ---------------------------------------------------------------------------

@mcp.tool()
def list_videos() -> str:
    """List all videos in the StudyTube library (id, title, status, upload date)."""
    # user_id=1: auth is not yet implemented; all videos belong to the seed user.
    with Session(engine) as session:
        videos = get_videos_for_user(session, user_id=1)

    payload = [
        {
            "id": v.id,
            "title": v.title,
            "status": v.status,
            "created_at": v.created_at.isoformat(),
        }
        for v in videos
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def get_transcript(video_id: int) -> str:
    """Return the full transcript text for a video.

    Args:
        video_id: The ID of the video.
    """
    with Session(engine) as session:
        video = get_video_by_id(session, video_id)
        if video is None:
            return f"No video found with id={video_id}."

        transcript = session.exec(
            select(Transcript).where(Transcript.video_id == video_id)
        ).first()

    if transcript is None or not transcript.content:
        return f"No transcript available yet for video id={video_id}."

    payload = {
        "video_id": video_id,
        "video_title": video.title,
        "source": transcript.source,
        "transcript": transcript.content,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def ask_video(video_id: int, question: str, top_k: int = 3) -> str:
    """Ask a natural-language question about a video using RAG.

    Retrieves the most relevant transcript chunks then calls GPT-4o-mini
    to produce a grounded answer. Returns the answer, confidence level,
    and whether the answer was grounded in retrieved chunks.

    Args:
        video_id: The ID of the video to ask about.
        question: A natural-language question about the video content.
        top_k:    Number of transcript chunks to use as context (1–10, default 3).
    """
    with Session(engine) as session:
        video = get_video_by_id(session, video_id)
        if video is None:
            return f"No video found with id={video_id}."

        response = answer_question(session, video_id, question, top_k)

    payload = {
        "video_id": video_id,
        "question": response.question,
        "answer": response.answer,
        "confidence_level": response.confidence_level,
        "grounded": response.grounded,
        "top_k_used": response.top_k,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def summarize_video(video_id: int) -> str:
    """Generate a structured AI study summary for a video's transcript.

    Returns a paragraph summary, bullet-point key takeaways, and a list
    of important terms with brief definitions. Useful for lecture review
    and creating study notes.

    Requires: video must be transcribed. OPENAI_API_KEY must be set.

    Args:
        video_id: The ID of the video to summarize.
    """
    with Session(engine) as session:
        try:
            payload = summarize_video_for_study(session, video_id)
        except ValueError as exc:
            return str(exc)
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def quiz_video(video_id: int, num_questions: int = 5) -> str:
    """Generate a multiple-choice quiz grounded in a video's transcript.

    Samples transcript chunks evenly across the whole video (not just the
    most similar section) so questions cover the full lecture. Each question
    has 4 options and a correct answer. All questions are grounded — the LLM
    is instructed not to invent facts outside the transcript.

    Requires: video must be transcribed, chunked, and embedded.

    Args:
        video_id:      The ID of the video to quiz on.
        num_questions: How many questions to generate (default 5).
    """
    with Session(engine) as session:
        try:
            payload = generate_video_study_quiz(session, video_id, num_questions)
        except ValueError as exc:
            return str(exc)
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def find_relevant_video(topic: str) -> str:
    """Search across all videos to find which lectures are most relevant to a topic.

    Useful for study discovery: "which of my lectures covers X?"
    Runs a semantic similarity search against every video that has been
    embedded, then returns them ranked from most to least relevant.
    Only videos that have been through the full pipeline (chunked + embedded)
    can be searched — others are silently skipped.

    Args:
        topic: A topic, concept, or question to search for across all lectures.
               Examples: "JWT authentication", "database indexing", "recursion"
    """
    with Session(engine) as session:
        results = find_relevant_videos_for_topic(session, user_id=1, topic=topic)
    return json.dumps(results, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Entry point (used by `python -m app.mcp_server`)
# mcp dev uses the `mcp` instance directly — no __main__ block needed for it.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
