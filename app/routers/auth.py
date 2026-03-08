from fastapi import APIRouter

# auth router — full implementation will be added in a future step.
# Planned endpoints:
#   POST /auth/register  → create a new user account
#   POST /auth/login     → return a JWT access token
#   POST /auth/refresh   → refresh an expired token
#   GET  /auth/me        → return the current authenticated user

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.get("/")
def auth_placeholder():
    """
    Placeholder endpoint to confirm the auth router is mounted.
    This will be replaced with real register/login endpoints.
    """
    return {"message": "Auth router is ready. Implementation coming soon."}
