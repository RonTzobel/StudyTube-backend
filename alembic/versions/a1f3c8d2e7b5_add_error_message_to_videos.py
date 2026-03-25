"""add error_message to videos

Revision ID: a1f3c8d2e7b5
Revises: d3f7a1b2c9e4
Create Date: 2026-03-25

Context
-------
The Video model now carries an error_message column populated by the RQ worker
when the processing pipeline fails. This gives the frontend and the user a
clear description of what went wrong instead of a bare "failed" status.

The column is nullable VARCHAR with no default:
  - NULL means the video has never failed (success or still in progress).
  - Non-NULL means a failure occurred; the value is the exception type + message,
    truncated to 500 characters by the worker.

Upgrade:   adds error_message to videos
Downgrade: removes it (data loss on any failed-video rows — only rollback
           intentionally in a dev environment)
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "a1f3c8d2e7b5"
down_revision = "d3f7a1b2c9e4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # IF NOT EXISTS makes this safe to run even if the column was added manually.
    op.execute(
        "ALTER TABLE videos ADD COLUMN IF NOT EXISTS error_message VARCHAR"
    )


def downgrade() -> None:
    op.drop_column("videos", "error_message")
