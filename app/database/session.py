from sqlmodel import SQLModel, Session, create_engine

from app.config.settings import settings

# Use psycopg (v3) when URL is postgresql:// or postgres:// (psycopg2 often missing on Python 3.14).
_url = settings.DATABASE_URL
if _url.startswith("postgresql://") and "+" not in _url.split(":", 1)[0]:
    _url = _url.replace("postgresql://", "postgresql+psycopg://", 1)
elif _url.startswith("postgres://") and "+" not in _url.split(":", 1)[0]:
    _url = _url.replace("postgres://", "postgresql+psycopg://", 1)

# create_engine sets up the connection pool to PostgreSQL.
# echo=True logs every SQL query — useful during development, turn off in production.
engine = create_engine(_url, echo=settings.DEBUG)


def create_db_and_tables() -> None:
    """
    Creates all database tables based on SQLModel model definitions.

    Call this once at application startup (see main.py lifespan).
    In production you would use Alembic migrations instead, but this
    is fine for the early development phase.
    """
    SQLModel.metadata.create_all(engine)


def get_session():
    """
    FastAPI dependency that yields a database session per request.

    Usage in a router:
        from fastapi import Depends
        from app.database.session import get_session

        @router.get("/something")
        def read_something(session: Session = Depends(get_session)):
            ...
    """
    with Session(engine) as session:
        yield session
