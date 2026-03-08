from datetime import datetime

from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    """Data required to register a new user."""

    email: EmailStr
    username: str
    password: str  # plain text — will be hashed in the service layer


class UserRead(BaseModel):
    """What the API returns when representing a user. Never expose the password."""

    id: int
    email: str
    username: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True  # allows building from SQLModel/ORM objects
