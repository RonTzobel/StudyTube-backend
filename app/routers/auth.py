"""
Auth router — email/password auth + Google OAuth.

Final public paths (router prefix="/auth", mounted under /api/v1 in main.py):
  POST /api/v1/auth/register
  POST /api/v1/auth/login
  GET  /api/v1/auth/me
  GET  /api/v1/auth/google/login
  GET  /api/v1/auth/google/callback
"""

import logging
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from sqlmodel import Session

from app.config.settings import settings
from app.core.dependencies import get_current_user
from app.core.security import create_access_token, verify_password
from app.database.session import get_session
from app.models.user import User
from app.schemas.auth import AuthResponse, LoginRequest, RegisterRequest, UserPublic
from app.services.user_service import create_user, get_or_create_google_user, get_user_by_email

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])

# ---------------------------------------------------------------------------
# Google OAuth constants
# ---------------------------------------------------------------------------

_GOOGLE_AUTH_URL     = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL    = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


# ---------------------------------------------------------------------------
# Local auth — register
# ---------------------------------------------------------------------------

@router.post("/register", response_model=AuthResponse, status_code=201)
def register(request: RegisterRequest, db: Session = Depends(get_session)):
    """
    Create a new account and return a JWT.

    Request body:
        { "full_name": "...", "email": "...", "password": "..." }

    Returns:
        { access_token, token_type, user: { id, email, username, is_active } }
    """
    if get_user_by_email(db, request.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists.",
        )

    user = create_user(
        db,
        email=request.email,
        username=request.full_name,
        password=request.password,
    )

    token = create_access_token(user.id)
    return AuthResponse(
        access_token=token,
        user=UserPublic.model_validate(user),
    )


# ---------------------------------------------------------------------------
# Local auth — login
# ---------------------------------------------------------------------------

@router.post("/login", response_model=AuthResponse)
def login(request: LoginRequest, db: Session = Depends(get_session)):
    """
    Verify email + password and return a JWT.

    Request body:
        { "email": "...", "password": "..." }

    Returns:
        { access_token, token_type, user: { id, email, username, is_active } }
    """
    user = get_user_by_email(db, request.email)
    if user is None or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated.",
        )

    token = create_access_token(user.id)
    return AuthResponse(
        access_token=token,
        user=UserPublic.model_validate(user),
    )


# ---------------------------------------------------------------------------
# Local auth — current user
# ---------------------------------------------------------------------------

@router.get("/me", response_model=UserPublic)
def me(current_user: User = Depends(get_current_user)):
    """
    Return the authenticated user's profile.

    Requires: Authorization: Bearer <token>
    """
    return UserPublic.model_validate(current_user)


# ---------------------------------------------------------------------------
# Google OAuth — login entry point
# ---------------------------------------------------------------------------

@router.get("/google/login")
def google_login():
    """
    Redirect the browser to Google's OAuth consent screen.

    Logs config status to confirm the .env values are loaded correctly.
    """
    # --- Temporary debug logging ---
    logger.warning(
        "Google OAuth config | GOOGLE_CLIENT_ID loaded: %s | "
        "GOOGLE_CLIENT_SECRET loaded: %s | GOOGLE_CLIENT_SECRET length: %d | "
        "GOOGLE_REDIRECT_URI: %s",
        "yes" if settings.GOOGLE_CLIENT_ID else "NO — missing from .env",
        "yes" if settings.GOOGLE_CLIENT_SECRET else "NO — missing from .env",
        len(settings.GOOGLE_CLIENT_SECRET),
        settings.GOOGLE_REDIRECT_URI,
    )
    # --- End debug logging ---

    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_CLIENT_ID is not configured. Add it to your .env file.",
        )

    params = {
        "client_id":     settings.GOOGLE_CLIENT_ID,
        "redirect_uri":  settings.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope":         "openid email profile",
        "access_type":   "offline",
    }
    return RedirectResponse(f"{_GOOGLE_AUTH_URL}?{urlencode(params)}")


# ---------------------------------------------------------------------------
# Google OAuth — callback
# ---------------------------------------------------------------------------

@router.get("/google/callback")
def google_callback(
    code: str = None,
    error: str = None,
    db: Session = Depends(get_session),
):
    """
    Google redirects here after the user grants (or denies) access.

    Steps:
      1. Check for OAuth errors.
      2. Exchange the authorization code for an access token.
      3. Fetch the user's profile from Google.
      4. Find or create an internal DB user.
      5. Issue the same app JWT as local login.
      6. Redirect the browser to the frontend /auth/callback page with
         the token in the query string — the frontend stores it identically
         to a local login token.
    """
    # Step 1 — handle OAuth errors (e.g. user clicked "Cancel")
    if error:
        raise HTTPException(status_code=400, detail=f"Google OAuth error: {error}")
    if not code:
        raise HTTPException(
            status_code=400,
            detail="No authorization code received from Google.",
        )

    # Step 2 — exchange authorization code for Google access token
    token_response = httpx.post(
        _GOOGLE_TOKEN_URL,
        data={
            "code":          code,
            "client_id":     settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "redirect_uri":  settings.GOOGLE_REDIRECT_URI,
            "grant_type":    "authorization_code",
        },
    )

    # --- Temporary debug logging for token exchange errors ---
    if token_response.status_code != 200:
        logger.warning(
            "Google token exchange failed | status=%d | body=%s",
            token_response.status_code,
            token_response.text,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Failed to exchange authorization code with Google: {token_response.text}",
        )
    # --- End debug logging ---

    google_access_token = token_response.json().get("access_token")
    if not google_access_token:
        raise HTTPException(
            status_code=400,
            detail="Google token response did not include an access token.",
        )

    # Step 3 — fetch user profile from Google
    userinfo_response = httpx.get(
        _GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {google_access_token}"},
    )
    if userinfo_response.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail="Failed to fetch user profile from Google.",
        )

    profile = userinfo_response.json()
    email = profile.get("email")
    if not email:
        raise HTTPException(
            status_code=400,
            detail="Google did not provide an email address. Cannot create account.",
        )

    # Step 4 — find or create internal DB user
    # If this email already belongs to a local-auth user, we reuse that row —
    # both login methods converge on the same DB identity.
    user = get_or_create_google_user(
        db,
        email=email,
        name=profile.get("name"),
    )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated.",
        )

    # Step 5 — issue the SAME internal JWT as local login
    app_token = create_access_token(user.id)

    # Step 6 — redirect the browser to the frontend callback page.
    # The frontend /auth/callback route reads ?token=, stores it in
    # localStorage as st_token, then redirects to the app home.
    # This is identical to what the frontend does after local login/register,
    # just delivered via URL redirect instead of a fetch() response body.
    redirect_url = f"{settings.FRONTEND_URL}/auth/callback?token={app_token}"
    return RedirectResponse(redirect_url)
