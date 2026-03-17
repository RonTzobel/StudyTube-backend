"""
Chat router — persistent tutor chat session endpoints.

All endpoints require a valid Bearer JWT.
Session ownership is enforced via user_id from the token.

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

from app.core.dependencies import get_current_user
from app.database.session import get_session
from app.models.user import User
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


@router.post("/sessions", response_model=ChatSessionResponse, status_code=201)
def create_session(
    request: CreateChatSessionRequest,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new tutor chat session for a video the caller owns.

    Request body:
      { "video_id": 1, "title": "My study session" }
    """
    try:
        return create_chat_session(
            db, user_id=current_user.id, video_id=request.video_id, title=request.title
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/sessions", response_model=List[ChatSessionResponse])
def list_sessions(
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """List all chat sessions for the authenticated user, newest first."""
    return list_chat_sessions_for_user(db, user_id=current_user.id)


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
def get_session_by_id(
    session_id: int,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Get a single chat session. Returns 404 if not found or not owned."""
    chat_session = get_chat_session_for_user(db, session_id, user_id=current_user.id)
    if chat_session is None:
        raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found.")
    return chat_session


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessageResponse])
def list_messages(
    session_id: int,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Return all messages for a session in conversation order (oldest first)."""
    chat_session = get_chat_session_for_user(db, session_id, user_id=current_user.id)
    if chat_session is None:
        raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found.")
    return get_chat_messages(db, session_id, user_id=current_user.id)


@router.post("/sessions/{session_id}/messages", response_model=SendMessageResponse)
def send_message(
    session_id: int,
    request: SendMessageRequest,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Send a message in a chat session and receive the tutor's reply.

    Request body:
      { "message": "What is JWT authentication?", "question_mode": "lecture" }
    """
    try:
        user_msg, assistant_msg, answer_source = send_chat_message(
            db,
            session_id=session_id,
            user_id=current_user.id,
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
    current_user: User = Depends(get_current_user),
):
    """Delete a chat session and all its messages. Returns 204 on success."""
    deleted = delete_chat_session(db, session_id, user_id=current_user.id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found.")
