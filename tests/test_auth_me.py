"""
tests/test_auth_me.py — focused tests for GET /api/v1/auth/me

These tests cover every distinct auth failure path plus the happy path.
The DB layer is mocked via FastAPI's dependency_override mechanism so the
tests run without a live PostgreSQL connection.

Covered cases:
  1. No Authorization header          → 401  (was 403 before the fix)
  2. Wrong scheme (Basic ...)          → 401  (was 403 before the fix)
  3. Garbage token string              → 401
  4. Expired JWT                       → 401
  5. JWT with null "sub" claim         → 401  (was 500 before the fix)
  6. Valid token, user not in DB       → 401
  7. Valid token, inactive user        → 401
  8. Valid token, active user          → 200 with correct payload
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from jose import jwt

from app.core.security import create_access_token
from app.database.session import get_session
from app.main import app
from app.models.user import User

ME = "/api/v1/auth/me"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _mock_session_returning(user_or_none):
    """
    Return a get_session override that makes session.get() return the given value.
    This is sufficient for get_user_by_id which calls session.get(User, id).
    """
    def _override():
        mock = MagicMock()
        mock.get.return_value = user_or_none
        yield mock
    return _override


def _active_user(user_id: int = 42) -> User:
    return User(
        id=user_id,
        email="test@example.com",
        username="testuser",
        hashed_password="irrelevant",
        is_active=True,
    )


def _inactive_user(user_id: int = 42) -> User:
    return User(
        id=user_id,
        email="inactive@example.com",
        username="inactiveuser",
        hashed_password="irrelevant",
        is_active=False,
    )


# ---------------------------------------------------------------------------
# Unauthenticated / bad token cases — no DB interaction needed
# ---------------------------------------------------------------------------

def test_me_no_authorization_header():
    """Missing Authorization header must return 401, not 403."""
    with TestClient(app) as client:
        response = client.get(ME)
    assert response.status_code == 401


def test_me_wrong_scheme():
    """Non-Bearer Authorization scheme must return 401, not 403."""
    with TestClient(app) as client:
        response = client.get(ME, headers={"Authorization": "Basic dXNlcjpwYXNz"})
    assert response.status_code == 401


def test_me_garbage_token():
    """A token string that is not a valid JWT must return 401."""
    with TestClient(app) as client:
        response = client.get(ME, headers=_bearer("this.is.not.a.jwt"))
    assert response.status_code == 401


def test_me_expired_token():
    """A syntactically valid but expired JWT must return 401."""
    expired = jwt.encode(
        {"sub": "1", "exp": datetime(2000, 1, 1, tzinfo=timezone.utc)},
        "changeme",
        algorithm="HS256",
    )
    with TestClient(app) as client:
        response = client.get(ME, headers=_bearer(expired))
    assert response.status_code == 401


def test_me_null_sub_claim():
    """
    JWT with a null 'sub' claim must return 401, not 500.
    Previously this caused int(None) → TypeError → unhandled 500.
    """
    bad_token = jwt.encode(
        {"sub": None, "exp": datetime(2099, 1, 1, tzinfo=timezone.utc)},
        "changeme",
        algorithm="HS256",
    )
    with TestClient(app) as client:
        response = client.get(ME, headers=_bearer(bad_token))
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Valid JWT, DB-dependent cases — mock the session
# ---------------------------------------------------------------------------

def test_me_user_not_in_db():
    """Valid token for a user_id that no longer exists in DB must return 401."""
    token = create_access_token(9999)
    app.dependency_overrides[get_session] = _mock_session_returning(None)
    try:
        with TestClient(app) as client:
            response = client.get(ME, headers=_bearer(token))
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 401


def test_me_inactive_user():
    """Valid token for a deactivated user must return 401."""
    user = _inactive_user(user_id=7)
    token = create_access_token(7)
    app.dependency_overrides[get_session] = _mock_session_returning(user)
    try:
        with TestClient(app) as client:
            response = client.get(ME, headers=_bearer(token))
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 401


def test_me_valid_token_and_user():
    """Valid token for an active user must return 200 with correct user fields."""
    user = _active_user(user_id=42)
    token = create_access_token(42)
    app.dependency_overrides[get_session] = _mock_session_returning(user)
    try:
        with TestClient(app) as client:
            response = client.get(ME, headers=_bearer(token))
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 42
    assert body["email"] == "test@example.com"
    assert body["username"] == "testuser"
    assert body["is_active"] is True
    # hashed_password must never appear in the response
    assert "hashed_password" not in body
    assert "password" not in body
