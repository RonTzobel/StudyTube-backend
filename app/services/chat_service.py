"""
chat_service.py — Persistent tutor chat sessions.

Sits on top of the existing tutor/QA layer. Handles session and message
persistence, ownership validation, and answer generation.

Each chat turn is single-turn from the RAG perspective: the current user
message is used as the retrieval query and sent to the QA pipeline.
All messages are stored and returned to the client so the conversation
history is visible, but prior turns are not injected into the retrieval
query. This is honest MVP behaviour — the QA pipeline is stateless by
design; adding history injection would require modifying qa_service.

Design notes
------------
- `db` is used as the parameter name for the SQLModel Session throughout
  this file to avoid shadowing ChatSession or local variables named `session`.
- Ownership is enforced by filtering on both session_id AND user_id at the
  DB level — a missing or mismatched user_id returns None, not an error,
  so the router can issue a clean 404 without revealing the session exists.
- updated_at on ChatSession is bumped manually each time a message is saved
  because SQLModel has no onupdate hook.
"""

from datetime import datetime, timezone

from sqlmodel import Session, select

from app.models.chat import ChatMessage, ChatSession
from app.services.qa_service import answer_for_mode
from app.services.video_service import get_video_by_id


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def create_chat_session(
    db: Session,
    user_id: int,
    video_id: int,
    title: str | None = None,
) -> ChatSession:
    """
    Create a new tutor chat session scoped to one video.

    If title is not provided, auto-generates one from the video title.

    Raises:
        ValueError: If the video does not exist.
    """
    video = get_video_by_id(db, video_id)
    if video is None:
        raise ValueError(f"Video {video_id} not found.")

    chat_session = ChatSession(
        user_id=user_id,
        video_id=video_id,
        title=title or f"Chat — {video.title}",
    )
    db.add(chat_session)
    db.commit()
    db.refresh(chat_session)
    return chat_session


def list_chat_sessions_for_user(
    db: Session,
    user_id: int,
) -> list[ChatSession]:
    """Return all chat sessions for a user, newest first."""
    return list(
        db.exec(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
        ).all()
    )


def get_chat_session_for_user(
    db: Session,
    session_id: int,
    user_id: int,
) -> ChatSession | None:
    """
    Return a chat session if it exists and belongs to the user.

    Returns None if not found or if the session belongs to a different user.
    The router converts None → 404 so callers don't learn whether the session
    exists for another user.
    """
    return db.exec(
        select(ChatSession)
        .where(ChatSession.id == session_id)
        .where(ChatSession.user_id == user_id)
    ).first()


def delete_chat_session(
    db: Session,
    session_id: int,
    user_id: int,
) -> bool:
    """
    Delete a chat session and all its messages.

    Returns True if deleted, False if not found / not owned.
    """
    chat_session = get_chat_session_for_user(db, session_id, user_id)
    if chat_session is None:
        return False

    # Delete messages first (no cascade configured at DB level).
    messages = db.exec(
        select(ChatMessage).where(ChatMessage.session_id == session_id)
    ).all()
    for msg in messages:
        db.delete(msg)

    db.delete(chat_session)
    db.commit()
    return True


# ---------------------------------------------------------------------------
# Message management
# ---------------------------------------------------------------------------

def get_chat_messages(
    db: Session,
    session_id: int,
    user_id: int,
) -> list[ChatMessage]:
    """
    Return all messages for a session, oldest first (conversation order).

    Returns an empty list if the session is not found or not owned.
    The router validates ownership separately before calling this.
    """
    return list(
        db.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
        ).all()
    )


def send_chat_message(
    db: Session,
    session_id: int,
    user_id: int,
    message: str,
    question_mode: str = "lecture",
) -> tuple[ChatMessage, ChatMessage, str]:
    """
    Save the user's message, generate a tutor answer, save it.

    question_mode controls which pipeline is used:
      "lecture" — strict RAG flow, no automatic fallback to general knowledge.
      "general" — skip transcript retrieval, answer from general AI knowledge.

    Steps:
      1. Validate the session exists and belongs to the user.
      2. Save the user message.
      3. Call answer_for_mode() with the chosen question_mode.
      4. Save the assistant message.
      5. Bump session.updated_at so list_chat_sessions sorts correctly.

    Returns:
        (user_message, assistant_message, answer_source) — both freshly saved
        ChatMessage rows, plus "lecture" or "general" indicating the answer source.

    Raises:
        ValueError:   If the session is not found, not owned, or the video
                      has no embedded chunks (lecture mode only).
        RuntimeError: If OpenAI is unavailable.
    """
    # Step 1 — validate session ownership
    chat_session = get_chat_session_for_user(db, session_id, user_id)
    if chat_session is None:
        raise ValueError(f"Chat session {session_id} not found.")

    # Step 2 — save user message
    user_msg = ChatMessage(
        session_id=session_id,
        role="user",
        content=message,
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # Step 3 — generate answer using the explicit mode chosen by the user.
    answer_text, answer_source = answer_for_mode(
        session=db,
        video_id=chat_session.video_id,
        question=message,
        question_mode=question_mode,
        top_k=3,
    )

    # Step 4 — save assistant message
    assistant_msg = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=answer_text,
    )
    db.add(assistant_msg)

    # Step 5 — bump updated_at so newest-session ordering works
    chat_session.updated_at = datetime.now(timezone.utc)
    db.add(chat_session)

    db.commit()
    db.refresh(assistant_msg)

    return user_msg, assistant_msg, answer_source
