"""
tests/test_auth_register.py — POST /api/v1/auth/register

Tests the register endpoint end-to-end at the HTTP layer.
The DB is mocked via unittest.mock.patch on the service functions so no live
PostgreSQL connection is required.

Covered cases:
  1. Valid registration                          → 201, access_token present
  2. Password at exactly 72 bytes (boundary)    → 201 (must be accepted)
  3. Password 73 ASCII bytes                    → 422 (not 500)
  4. Password <=72 chars but >72 UTF-8 bytes    → 422 (byte check, not char check)
  5. Duplicate email                            → 409 (not confused with the above)

Why byte length matters:
  bcrypt operates on bytes. len(password) counts Unicode characters, not bytes.
  A Hebrew letter ("א") is 1 char but 2 bytes; an emoji is 1 char but 4 bytes.
  37 Hebrew chars = 74 bytes — would pass a naive len() check but fail bcrypt.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.models.user import User

REGISTER = "/api/v1/auth/register"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_user(user_id: int = 1, email: str = "alice@example.com") -> User:
    return User(
        id=user_id,
        email=email,
        username="Alice",
        hashed_password="$2b$12$irrelevant_for_register_tests",
        is_active=True,
    )


def _payload(
    full_name: str = "Alice",
    email: str = "alice@example.com",
    password: str = "SecurePass1!",
) -> dict:
    return {"full_name": full_name, "email": email, "password": password}


def _register(payload: dict) -> object:
    with TestClient(app) as client:
        return client.post(REGISTER, json=payload)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_register_valid():
    """Valid registration returns 201 with an access_token and safe user fields."""
    user = _make_user()
    with (
        patch("app.routers.auth.get_user_by_email", return_value=None),
        patch("app.routers.auth.create_user", return_value=user),
    ):
        response = _register(_payload())

    assert response.status_code == 201
    body = response.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert body["user"]["email"] == "alice@example.com"
    assert body["user"]["username"] == "Alice"
    assert body["user"]["is_active"] is True
    # Security: hashed_password must never appear in any auth response
    assert "hashed_password" not in body["user"]
    assert "password" not in body["user"]


def test_register_password_exactly_72_bytes():
    """Password of exactly 72 ASCII bytes (= 72 chars) is at the limit and must be accepted."""
    user = _make_user()
    with (
        patch("app.routers.auth.get_user_by_email", return_value=None),
        patch("app.routers.auth.create_user", return_value=user),
    ):
        response = _register(_payload(password="a" * 72))

    assert response.status_code == 201


# ---------------------------------------------------------------------------
# Password too long — must return 422, never 500
# ---------------------------------------------------------------------------

def test_register_password_73_ascii_bytes():
    """
    Password of 73 ASCII bytes (73 chars = 73 bytes) exceeds the bcrypt limit.
    Must return 422 from schema validation — not 500 from hash_password().
    No DB call should occur.
    """
    response = _register(_payload(password="a" * 73))

    assert response.status_code == 422
    errors = response.json()["detail"]
    # Pydantic v2 loc format: ["body", "password"]
    assert any("password" in str(e.get("loc", "")) for e in errors)
    # Error message must be user-friendly — bcrypt internals must not be exposed
    messages = " ".join(str(e.get("msg", "")) for e in errors).lower()
    assert "too long" in messages
    assert "72" not in messages


def test_register_password_too_long_multibyte():
    """
    Password that is <=72 characters but >72 UTF-8 bytes must also return 422.

    Each Hebrew letter (e.g. "א") is 2 bytes in UTF-8.
    37 Hebrew chars = 74 bytes — would pass a naive len(password) > 72 guard
    but must be caught by the correct byte-length check.
    """
    hebrew_password = "א" * 37   # 37 chars, 74 bytes
    assert len(hebrew_password) == 37           # char length looks fine
    assert len(hebrew_password.encode()) == 74  # byte length exceeds limit

    response = _register(_payload(password=hebrew_password))

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("password" in str(e.get("loc", "")) for e in errors)


def test_register_password_too_long_emoji():
    """
    Each emoji is 4 bytes in UTF-8.
    19 emoji chars = 76 bytes — 19 chars would pass a len()-only check.
    """
    emoji_password = "🔒" * 19   # 19 chars, 76 bytes
    assert len(emoji_password) == 19
    assert len(emoji_password.encode()) == 76

    response = _register(_payload(password=emoji_password))

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Duplicate email — must remain a 409, not be confused with password errors
# ---------------------------------------------------------------------------

def test_register_duplicate_email():
    """Registering with an already-taken email returns 409 Conflict."""
    existing = _make_user(email="alice@example.com")
    with patch("app.routers.auth.get_user_by_email", return_value=existing):
        response = _register(_payload())

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"].lower()


def test_register_duplicate_email_not_confused_with_password_error():
    """
    A duplicate-email rejection must NOT return 422.
    Ensures the two error paths are distinct and cannot be mistaken for each other.
    """
    existing = _make_user(email="alice@example.com")
    with patch("app.routers.auth.get_user_by_email", return_value=existing):
        response = _register(_payload())

    # Must be 409, never 422 (which is for validation errors like bad password)
    assert response.status_code != 422
    assert response.status_code == 409
