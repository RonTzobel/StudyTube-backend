from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    """
    Represents a registered user.

    This is intentionally minimal for now.
    Future additions: hashed_password, role, profile picture, etc.
    """

    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True, nullable=False)
    username: str = Field(unique=True, index=True, nullable=False)

    # Placeholder for hashed password — auth will be implemented later
    hashed_password: str = Field(nullable=False)

    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
