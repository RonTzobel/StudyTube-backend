"""
alembic/env.py — Alembic migration environment for StudyTube backend.

Design decisions:
  - The database URL is taken from app.database.session.engine, which reads
    DATABASE_URL from settings and normalises postgres:// → postgresql+psycopg://.
    This is the single source of truth — no URL duplication.

  - All SQLModel table metadata is loaded by importing app.models, which already
    imports every model in one place (app/models/__init__.py). Alembic sees the
    complete schema without needing per-model imports here.

  - Both online (real connection) and offline (SQL script generation) modes are
    supported, matching the standard Alembic template.

Running from the project root (backend/):
    alembic upgrade head
    alembic current
    alembic history --verbose
    alembic revision --autogenerate -m "describe your change"
"""

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlmodel import SQLModel

# ---------------------------------------------------------------------------
# Make sure `app` is importable regardless of the working directory.
# Inside the Docker container, WORKDIR=/app so this is already on sys.path.
# Locally, running `alembic` from backend/ adds '.' via alembic.ini's
# prepend_sys_path — this line is a belt-and-suspenders fallback.
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ---------------------------------------------------------------------------
# Import the engine from the project.
# This also loads settings and normalises the DATABASE_URL automatically.
# ---------------------------------------------------------------------------
from app.database.session import engine  # noqa: E402

# ---------------------------------------------------------------------------
# Import all models so Alembic's --autogenerate can see every table.
# app.models.__init__ already imports User, Video, Transcript, TranscriptChunk,
# Summary, ChatSession, ChatMessage — all in one import.
# ---------------------------------------------------------------------------
import app.models  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Alembic Config object (gives access to values in alembic.ini).
# ---------------------------------------------------------------------------
config = context.config

# Apply Python logging configuration from alembic.ini, if present.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# This is the metadata Alembic compares against the live DB for --autogenerate.
target_metadata = SQLModel.metadata


# ---------------------------------------------------------------------------
# Offline mode — generates a .sql script without connecting to the DB.
# Usage: alembic upgrade head --sql > migration.sql
# ---------------------------------------------------------------------------
def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (no live DB connection needed)."""
    context.configure(
        url=engine.url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online mode — connects to the live DB and applies migrations.
# This is the standard mode used by `alembic upgrade head`.
# ---------------------------------------------------------------------------
def run_migrations_online() -> None:
    """Run migrations in 'online' mode (live DB connection)."""
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
