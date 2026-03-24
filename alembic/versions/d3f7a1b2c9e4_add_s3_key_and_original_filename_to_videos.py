"""add s3_key and original_filename to videos

Revision ID: d3f7a1b2c9e4
Revises:
Create Date: 2026-03-24

Context
-------
The Video model in app/models/video.py defines two columns that were added
after the initial table creation:

    s3_key           Optional[str]  — S3 object key, e.g. "videos/7/abc.mp4"
    original_filename Optional[str] — filename as submitted by the uploader

Because SQLModel.metadata.create_all() only creates NEW tables and never
alters existing ones, these columns are absent on any database instance whose
`videos` table was created before the model was updated.

This migration adds them as nullable VARCHAR columns with no default so that:
  - All existing rows remain valid (NULL is the correct value for legacy rows
    that were uploaded before S3 storage was introduced).
  - The migration is safe to run whether or not the columns already exist
    (uses IF NOT EXISTS — idempotent against manual SQL fixes).

Upgrade:   adds s3_key, original_filename to videos
Downgrade: removes those columns (data loss — only roll back intentionally)
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "d3f7a1b2c9e4"
down_revision = None      # first migration — no parent
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Using raw SQL with IF NOT EXISTS so this migration is safe to apply
    # even if the columns were already added manually via the raw SQL file
    # at migrations/add_video_s3_fields.sql.
    #
    # op.add_column would raise ProgrammingError: column already exists
    # in that case, making the migration non-rerunnable. IF NOT EXISTS avoids
    # that entirely. This is PostgreSQL-specific syntax; this project targets
    # Aiven PostgreSQL exclusively so it is always available.
    op.execute(
        "ALTER TABLE videos ADD COLUMN IF NOT EXISTS s3_key VARCHAR"
    )
    op.execute(
        "ALTER TABLE videos ADD COLUMN IF NOT EXISTS original_filename VARCHAR"
    )


def downgrade() -> None:
    # WARNING: dropping these columns destroys data.
    # Only run downgrade intentionally (not as part of an emergency rollback
    # unless you are certain the columns are empty or data loss is acceptable).
    op.drop_column("videos", "original_filename")
    op.drop_column("videos", "s3_key")
