from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config.settings import settings
from app.database.session import create_db_and_tables
from app.routers import auth, health, videos


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup and once at shutdown.

    Startup: create database tables if they don't exist yet.
    This is fine for development. In production, use Alembic migrations.
    """
    create_db_and_tables()
    yield
    # Shutdown logic can go here if needed (e.g. close connections)


app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered learning assistant for videos.",
    version="0.1.0",
    lifespan=lifespan,
)

# --- Register routers ---
# Each router handles a distinct area of the API.
app.include_router(health.router)
app.include_router(auth.router, prefix="/api/v1")
app.include_router(videos.router, prefix="/api/v1")


@app.get("/")
def root():
    """Root endpoint — confirms the API is reachable."""
    return {"message": f"Welcome to {settings.APP_NAME}"}
