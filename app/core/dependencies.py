"""
dependencies.py — shared FastAPI dependency functions.

Import and use with Depends() in routers.
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel import Session

from app.core.security import decode_access_token
from app.database.session import get_session
from app.models.user import User
from app.services.user_service import get_user_by_id

# auto_error=False so we control the response code ourselves.
# FastAPI's default (auto_error=True) raises HTTP 403 for a missing or
# malformed Authorization header. That is technically valid per RFC 7235,
# but it conflicts with the frontend's expectation that auth failures are
# always 401.  By setting auto_error=False, credentials is None when the
# header is absent/malformed and we raise 401 explicitly below.
_bearer = HTTPBearer(auto_error=False)

_401 = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated.",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_session),
) -> User:
    """
    Validate the Bearer JWT and return the authenticated User.

    Raises 401 (never 403) for every auth failure so the frontend can
    handle them uniformly:
      - Authorization header missing or not a Bearer scheme
      - Token invalid, malformed, or expired
      - Token references a user that no longer exists or is inactive
    """
    if credentials is None:
        raise _401

    user_id = decode_access_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_id(db, user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user
