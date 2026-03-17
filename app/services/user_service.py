import secrets

from sqlmodel import Session, select

from app.core.security import hash_password
from app.models.user import User


def get_user_by_email(session: Session, email: str) -> User | None:
    """Look up a user by their email address."""
    return session.exec(select(User).where(User.email == email)).first()


def get_user_by_id(session: Session, user_id: int) -> User | None:
    """Look up a user by their primary key."""
    return session.get(User, user_id)


def _get_user_by_username(session: Session, username: str) -> User | None:
    return session.exec(select(User).where(User.username == username)).first()


def get_or_create_google_user(
    session: Session,
    email: str,
    name: str | None,
) -> User:
    """
    Find an existing user by email or create a new one for Google OAuth.

    If the email already exists (local-auth or a previous Google login),
    that user row is returned as-is — no duplicate accounts are created and
    no existing fields are overwritten.

    For brand-new Google users:
      - username: Google display name when available and not already taken;
        falls back to the email address (which is always unique).
      - hashed_password: bcrypt hash of a 64-char random secret.
        Nobody knows this password, so local verify_password calls will
        always fail safely — the user must authenticate via Google.

    Returns:
        The existing or newly created User.
    """
    user = get_user_by_email(session, email)
    if user:
        return user

    # Choose a username — prefer Google name, fall back to email if taken
    preferred = (name or "").strip() or email
    username = preferred if not _get_user_by_username(session, preferred) else email

    return create_user(
        session,
        email=email,
        username=username,
        password=secrets.token_hex(16),  # 32 hex chars = 32 bytes, safely under bcrypt's 72-byte limit
    )


def create_user(session: Session, email: str, username: str, password: str) -> User:
    """
    Create and persist a new user with a bcrypt-hashed password.

    Args:
        session:  DB session.
        email:    The user's email address (must be unique).
        username: Display name / handle (must be unique).
        password: Plain-text password — hashed here before storing.

    Returns:
        The newly created User row.
    """
    user = User(
        email=email,
        username=username,
        hashed_password=hash_password(password),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user
