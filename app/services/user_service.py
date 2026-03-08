from sqlmodel import Session, select

from app.models.user import User
from app.schemas.user import UserCreate


def get_user_by_email(session: Session, email: str) -> User | None:
    """Look up a user by their email address."""
    return session.exec(select(User).where(User.email == email)).first()


def get_user_by_id(session: Session, user_id: int) -> User | None:
    """Look up a user by their primary key."""
    return session.get(User, user_id)


def create_user(session: Session, user_data: UserCreate) -> User:
    """
    Create and persist a new user.

    NOTE: Password hashing will be added here once auth is implemented.
    For now we store a placeholder to satisfy the non-null constraint.
    """
    # TODO: replace with proper bcrypt hashing when auth is implemented
    hashed_password = f"hashed_{user_data.password}"

    user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user
