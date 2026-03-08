from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    """
    Simple health check endpoint.

    Used by load balancers, Docker health checks, and monitoring tools
    to verify the API is running. Returns a fixed response — no DB call needed.
    """
    return {"status": "ok", "service": "StudyTube API"}
