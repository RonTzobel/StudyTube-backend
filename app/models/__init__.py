# models/
# SQLModel table definitions — these map directly to database tables.
# Each file represents one entity (User, Video, Transcript, etc.).
# Keep models lean: no business logic here, only field definitions.
#
# Import EVERY model here so that:
#   1. SQLModel.metadata has a complete picture of all tables.
#   2. Any process (API server, RQ worker, Alembic) that does
#      `import app.models` gets all FK targets registered — without
#      this, SQLAlchemy raises NoReferencedTableError on flush/commit
#      for any FK whose target table was not explicitly imported.
from app.models.user import User  # noqa: F401
from app.models.video import Video  # noqa: F401
from app.models.transcript import Transcript  # noqa: F401
from app.models.chunk import TranscriptChunk  # noqa: F401
from app.models.summary import Summary  # noqa: F401
from app.models.chat import ChatSession, ChatMessage  # noqa: F401
