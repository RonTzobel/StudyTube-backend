from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class ChatSession(SQLModel, table=True):
    """
    A persistent tutor conversation scoped to a single video.

    One session belongs to one user and one video. Multiple sessions
    can exist for the same video (e.g. a student revisits a lecture).

    The title is optional — if not supplied by the user it is
    auto-generated from the video title in the service layer.

    updated_at is set at creation and bumped manually each time a new
    message is saved. SQLModel has no onupdate hook, so the service
    handles this explicitly.
    """

    __tablename__ = "chat_sessions"

    id: Optional[int] = Field(default=None, primary_key=True)

    user_id: int = Field(foreign_key="users.id", nullable=False, index=True)
    video_id: int = Field(foreign_key="videos.id", nullable=False, index=True)

    title: Optional[str] = Field(default=None)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatMessage(SQLModel, table=True):
    """
    A single message inside a ChatSession.

    role is either "user" (the student's question) or "assistant"
    (the tutor's grounded answer). They are always saved in pairs:
    the user message first, then the assistant message immediately after.
    """

    __tablename__ = "chat_messages"

    id: Optional[int] = Field(default=None, primary_key=True)

    session_id: int = Field(foreign_key="chat_sessions.id", nullable=False, index=True)

    # "user" or "assistant"
    role: str = Field(nullable=False)

    content: str = Field(nullable=False)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
