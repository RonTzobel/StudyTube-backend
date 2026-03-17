"""
Schemas for /auth endpoints (register, login, Google OAuth).

Kept separate from app/schemas/user.py because the auth API shapes
differ from the generic user CRUD shapes.
"""

from pydantic import BaseModel, EmailStr


class RegisterRequest(BaseModel):
    """POST /auth/register"""
    full_name: str          # stored as username in the User table
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    """POST /auth/login"""
    email: EmailStr
    password: str


class UserPublic(BaseModel):
    """User fields safe to return in any auth response."""
    id: int
    email: str
    username: str
    is_active: bool

    class Config:
        from_attributes = True


class AuthResponse(BaseModel):
    """
    Returned by /register, /login, and (later) /auth/google/callback.

    Both Google OAuth and email/password login produce the same shape
    so the frontend can handle them identically.
    """
    access_token: str
    token_type: str = "bearer"
    user: UserPublic
