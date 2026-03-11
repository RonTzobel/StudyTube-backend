# models/
# SQLModel table definitions — these map directly to database tables.
# Each file represents one entity (User, Video, Transcript, etc.).
# Keep models lean: no business logic here, only field definitions.
#
# Import every model here so SQLModel.metadata.create_all() in session.py
# can see them and create their tables on startup.
from app.models.chunk import TranscriptChunk  # noqa: F401
from app.models.summary import Summary  # noqa: F401
