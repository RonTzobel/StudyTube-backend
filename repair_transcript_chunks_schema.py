"""
repair_transcript_chunks_schema.py
===================================
One-time (but idempotent) script that brings the PostgreSQL
transcript_chunks table in sync with the current TranscriptChunk
SQLModel definition.

Why this is needed
------------------
SQLModel.metadata.create_all() only CREATES tables that are missing.
It never adds new columns to an existing table.  When embedding and
is_embedded were added to the TranscriptChunk model after the table
had already been created, the real PostgreSQL table was left behind.

This script fixes that with ALTER TABLE … ADD COLUMN IF NOT EXISTS,
which is safe to run any number of times — it does nothing if the
column already exists.

Usage
-----
    cd C:/Users/Ronzo/StudyTube/backend
    python repair_transcript_chunks_schema.py

What it repairs
---------------
Column          | PostgreSQL type              | SQLModel equivalent
----------------|------------------------------|---------------------
embedding       | TEXT                         | Optional[str]
is_embedded     | BOOLEAN NOT NULL DEFAULT FALSE | bool = False

Both columns match exactly what the current model declares.
"""

from sqlalchemy import text

# Reuse the engine that the rest of the app uses — reads DATABASE_URL
# from .env automatically via app/config/settings.py.
from app.database.session import engine

# ---------------------------------------------------------------------------
# Columns to add.
#
# Each entry is:
#   (column_name, column_definition)
#
# ADD COLUMN IF NOT EXISTS is a PostgreSQL 9.6+ feature — safe to run
# even if the column already exists (no error, no data loss).
# ---------------------------------------------------------------------------
COLUMNS_TO_ADD = [
    (
        "embedding",
        "TEXT",
        # Optional[str] in SQLModel → TEXT in PostgreSQL.
        # Stores the embedding vector as a JSON string, e.g. "[0.12, -0.34, ...]".
        # NULL means the chunk has not been embedded yet.
    ),
    (
        "is_embedded",
        "BOOLEAN NOT NULL DEFAULT FALSE",
        # bool = Field(default=False) in SQLModel → BOOLEAN NOT NULL DEFAULT FALSE.
        # Set to TRUE once the embedding has been generated and saved.
    ),
]


def repair() -> None:
    print("=== repair_transcript_chunks_schema.py ===\n")

    with engine.connect() as conn:
        for col_name, col_def, *_ in COLUMNS_TO_ADD:
            sql = (
                f"ALTER TABLE transcript_chunks "
                f"ADD COLUMN IF NOT EXISTS {col_name} {col_def};"
            )
            print(f"Running: {sql}")
            conn.execute(text(sql))
            print(f"  OK: column '{col_name}' is present.\n")

        conn.commit()

    print("=== Schema repair complete. ===")
    print("All required columns exist in transcript_chunks.")
    print("You can now restart uvicorn and use /chunk, /embed, and /search.\n")


if __name__ == "__main__":
    repair()
