"""
Schemas for /auth endpoints (register, login, Google OAuth).

Kept separate from app/schemas/user.py because the auth API shapes
differ from the generic user CRUD shapes.
"""

from pydantic import BaseModel, EmailStr, field_validator

# bcrypt silently truncates passwords longer than 72 bytes, which is a security
# risk (two different passwords can produce the same hash). passlib raises
# ValueError before hashing when the limit is exceeded — which is the 500 we
# are preventing here. Validation must use byte length, not character length,
# because a single Unicode character can be up to 4 bytes in UTF-8.
_BCRYPT_MAX_BYTES = 72


class RegisterRequest(BaseModel):
    """POST /auth/register"""
    full_name: str          # stored as username in the User table
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_fits_bcrypt(cls, v: str) -> str:
        # bcrypt operates on bytes and rejects inputs longer than 72 bytes.
        # We check encoded byte length — not character length — because a
        # single Unicode character can be up to 4 bytes in UTF-8.
        # The user-facing message is intentionally generic: there is no reason
        # to tell clients about the 72-byte bcrypt internals.
        if len(v.encode("utf-8")) > _BCRYPT_MAX_BYTES:
            raise ValueError("Password is too long. Please choose a shorter password.")
        return v


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
