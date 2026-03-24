from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session

from app.config.settings import settings
from app.database.session import create_db_and_tables, engine
from app.models.chunk import TranscriptChunk  # noqa: F401 — registers table with SQLModel
from app.models.summary import Summary  # noqa: F401 — registers table with SQLModel
from app.models.chat import ChatSession, ChatMessage  # noqa: F401 — registers tables with SQLModel
from app.models.user import User
from app.routers import auth, chat, health, tutor, videos


def _seed_default_user() -> None:
    """
    Create a placeholder user with id=1 if one does not exist.

    This is a TEMPORARY measure until JWT authentication is implemented.
    All uploaded videos are currently assigned to this user (user_id=1).

    Once real auth is in place:
      - Remove this function.
      - Replace the hardcoded user_id in the upload endpoint with the
        user id extracted from the JWT token.
    """
    with Session(engine) as session:
        if session.get(User, 1) is None:
            placeholder = User(
                email="default@studytube.dev",
                username="default",
                hashed_password="placeholder_not_a_real_password",
            )
            session.add(placeholder)
            session.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup and once at shutdown.

    Startup:
      1. Create database tables (if they don't exist yet).
         NOTE: create_db_and_tables() only creates NEW tables — it never alters
         existing ones. Schema changes to existing tables (add/drop columns,
         indexes, constraints) must be handled by Alembic migrations.
         Run `alembic upgrade head` before starting the app in production.
      2. Seed the placeholder user so foreign keys work before auth is live.
    """
    create_db_and_tables()
    _seed_default_user()
    yield
    # Shutdown logic can go here if needed (e.g. close connections)


app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered learning assistant for videos.",
    version="0.1.0",
    lifespan=lifespan,
)

# --- CORS ---
# Origins are controlled by CORS_ALLOWED_ORIGINS in .env so this list
# never needs to be changed in source code across environments.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# --- Register routers ---
# Each router handles a distinct area of the API.
app.include_router(health.router)
app.include_router(auth.router, prefix="/api/v1")
app.include_router(videos.router, prefix="/api/v1")
app.include_router(tutor.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")


@app.get("/")
def root():
    """Root endpoint — confirms the API is reachable."""
    return {"message": f"Welcome to {settings.APP_NAME}"}
