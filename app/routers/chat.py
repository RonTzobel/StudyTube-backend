"""
Chat router — persistent tutor chat session endpoints.

Follows the same structure as app/routers/tutor.py and app/routers/videos.py:
  - thin HTTP layer only
  - all logic delegated to chat_service
  - ValueError → 4xx, RuntimeError → 500

Endpoints
---------
  POST   /chat/sessions                         — create a new session
  GET    /chat/sessions                         — list the user's sessions
  GET    /chat/sessions/{session_id}            — get a single session
  GET    /chat/sessions/{session_id}/messages   — load conversation history
  POST   /chat/sessions/{session_id}/messages   — send a message, get reply
  DELETE /chat/sessions/{session_id}            — delete a session
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.database.session import get_session
from app.schemas.chat import (
    ChatMessageResponse,
    ChatSessionResponse,
    CreateChatSessionRequest,
    SendMessageRequest,
    SendMessageResponse,
)
from app.services.chat_service import (
    create_chat_session,
    delete_chat_session,
    get_chat_messages,
    get_chat_session_for_user,
    list_chat_sessions_for_user,
    send_chat_message,
)

router = APIRouter(prefix="/chat", tags=["Chat"])

# user_id=1: auth placeholder until JWT is implemented.
_CURRENT_USER_ID = 1


@router.post("/sessions", response_model=ChatSessionResponse, status_code=201)
def create_session(
    request: CreateChatSessionRequest,
    db: Session = Depends(get_session),
):
    """
    Create a new tutor chat session for a video.

    The session title is optional — if omitted it is auto-generated from
    the video title (e.g. "Chat — Lecture 3: Authentication").

    Request body:
      { "video_id": 1, "title": "My study session" }
    """
    try:
        return create_chat_session(
            db, user_id=_CURRENT_USER_ID, video_id=request.video_id, title=request.title
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/sessions", response_model=List[ChatSessionResponse])
def list_sessions(db: Session = Depends(get_session)):
    """
    List all chat sessions for the current user, newest first.

    Returns an empty list if no sessions exist.
    """
    return list_chat_sessions_for_user(db, user_id=_CURRENT_USER_ID)


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
def get_session_by_id(
    session_id: int,
    db: Session = Depends(get_session),
):
    """Get a single chat session by id."""
    chat_session = get_chat_session_for_user(db, session_id, user_id=_CURRENT_USER_ID)
    if chat_session is None:
        raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found.")
    return chat_session


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessageResponse])
def list_messages(
    session_id: int,
    db: Session = Depends(get_session),
):
    """
    Return all messages for a session in conversation order (oldest first).

    Returns an empty list if no messages have been sent yet.
    Raises 404 if the session does not exist or does not belong to the user.
    """
    chat_session = get_chat_session_for_user(db, session_id, user_id=_CURRENT_USER_ID)
    if chat_session is None:
        raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found.")
    return get_chat_messages(db, session_id, user_id=_CURRENT_USER_ID)


@router.post("/sessions/{session_id}/messages", response_model=SendMessageResponse)
def send_message(
    session_id: int,
    request: SendMessageRequest,
    db: Session = Depends(get_session),
):
    """
    Send a message in a chat session and receive the tutor's reply.

    The tutor answer is grounded in the video's transcript using the same
    RAG pipeline as POST /api/v1/videos/{video_id}/ask.

    Returns both the saved user message and the assistant reply so the
    client can render the full exchange in one round trip.

    Request body:
      { "message": "What is JWT authentication?" }

    Prerequisites:
      - The linked video must have been transcribed, chunked, and embedded.
      - OPENAI_API_KEY must be set in .env.
    """
    try:
        user_msg, assistant_msg, answer_source = send_chat_message(
            db,
            session_id=session_id,
            user_id=_CURRENT_USER_ID,
            message=request.message,
            question_mode=request.question_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return SendMessageResponse(
        user_message=ChatMessageResponse(
            id=user_msg.id,
            session_id=user_msg.session_id,
            role=user_msg.role,
            content=user_msg.content,
            created_at=user_msg.created_at,
        ),
        assistant_message=ChatMessageResponse(
            id=assistant_msg.id,
            session_id=assistant_msg.session_id,
            role=assistant_msg.role,
            content=assistant_msg.content,
            created_at=assistant_msg.created_at,
        ),
        answer_source=answer_source,
    )


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(
    session_id: int,
    db: Session = Depends(get_session),
):
    """
    Delete a chat session and all its messages.

    Returns 204 No Content on success.
    Raises 404 if the session does not exist or does not belong to the user.
    """
    deleted = delete_chat_session(db, session_id, user_id=_CURRENT_USER_ID)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found.")
