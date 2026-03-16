"""
Schemas for the /chat REST endpoints.

Kept minimal: only what the endpoints actually need.
"""

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class CreateChatSessionRequest(BaseModel):
    """POST /chat/sessions"""
    video_id: int
    title: Optional[str] = Field(default=None, description="Optional session title. Auto-generated from the video title if not provided.")


class SendMessageRequest(BaseModel):
    """POST /chat/sessions/{session_id}/messages"""
    message: str = Field(..., min_length=1, description="The student's question or message.")
    question_mode: str = Field(
        default="lecture",
        description=(
            "'lecture' to answer strictly from the video transcript (no fallback). "
            "'general' to skip transcript retrieval and answer from general AI knowledge."
        ),
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class ChatSessionResponse(BaseModel):
    """A single chat session."""
    id: int
    user_id: int
    video_id: int
    title: Optional[str]
    created_at: datetime
    updated_at: datetime


class ChatMessageResponse(BaseModel):
    """A single message (user or assistant)."""
    id: int
    session_id: int
    role: str            # "user" | "assistant"
    content: str
    created_at: datetime


class SendMessageResponse(BaseModel):
    """
    Returned by POST /chat/sessions/{session_id}/messages.

    Contains both the saved user message and the assistant's reply so the
    client can render both in one round trip.

    answer_source: "lecture" if the answer came from the video transcript via
    RAG; "general" if no relevant transcript chunks were found and OpenAI was
    called in general-knowledge mode instead.
    """
    user_message: ChatMessageResponse
    assistant_message: ChatMessageResponse
    answer_source: str  # "lecture" | "general"
