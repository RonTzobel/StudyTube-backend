import json
from typing import List

from sqlmodel import Session, select

from app.models.chunk import TranscriptChunk


# ---------------------------------------------------------------------------
# Local embedding model — loaded once at module import time.
#
# Model: all-MiniLM-L6-v2
#   - 384-dimensional vectors
#   - ~80 MB download, cached in ~/.cache/torch/sentence_transformers
#   - Runs on CPU; no GPU required
#   - Optimised for semantic similarity — exactly what RAG retrieval needs
#   - torch is already in the venv (pulled in by openai-whisper), so no
#     extra heavyweight dependency is added
#
# The model is loaded here (module level) so the ~1-2 second load cost
# is paid once at startup, not on every embed request.
#
# To swap to OpenAI embeddings later, delete the _model lines and replace
# embed_text() with an openai.embeddings.create() call. Nothing else changes.
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer as _ST
    _model = _ST("all-MiniLM-L6-v2")
except Exception:
    # sentence-transformers not installed — server still starts, embed
    # endpoint will raise a clear RuntimeError at call time.
    _model = None


def embed_text(text: str) -> List[float]:
    """
    Generate a 384-dimensional semantic embedding for a piece of text.

    Uses the locally-running all-MiniLM-L6-v2 model via sentence-transformers.
    No API key or network call is needed after the first run (model is cached).

    Args:
        text: The chunk text to embed.

    Returns:
        A list of 384 floats representing the semantic embedding vector.

    Raises:
        RuntimeError: If sentence-transformers is not installed.
    """
    if _model is None:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )
    vector = _model.encode(text, convert_to_numpy=True)
    return vector.tolist()


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_unembedded_chunks(session: Session, video_id: int) -> List[TranscriptChunk]:
    """
    Return all chunks for a video that have not been embedded yet.

    Uses the is_embedded flag rather than checking embedding IS NULL so
    the query stays fast and the intent is explicit.
    """
    return list(
        session.exec(
            select(TranscriptChunk)
            .where(TranscriptChunk.video_id == video_id)
            .where(TranscriptChunk.is_embedded == False)  # noqa: E712
            .order_by(TranscriptChunk.chunk_index)
        ).all()
    )


def save_embedding(session: Session, chunk: TranscriptChunk, vector: List[float]) -> TranscriptChunk:
    """
    Persist an embedding vector onto a single chunk row.

    The vector is serialised to a JSON string for storage.
    is_embedded is set to True to mark the chunk as done.

    Args:
        session:  The database session injected by FastAPI.
        chunk:    The TranscriptChunk ORM object to update.
        vector:   The embedding vector returned by embed_text().

    Returns:
        The updated, refreshed TranscriptChunk object.
    """
    chunk.embedding = json.dumps(vector)
    chunk.is_embedded = True
    session.add(chunk)
    session.commit()
    session.refresh(chunk)
    return chunk


def embed_all_chunks(session: Session, video_id: int) -> List[TranscriptChunk]:
    """
    Embed every un-embedded chunk for a video and persist the results.

    Skips chunks that already have is_embedded=True, so this is safe to
    call more than once (idempotent when all chunks are already embedded).

    Args:
        session:   The database session injected by FastAPI.
        video_id:  The video whose chunks should be embedded.

    Returns:
        The full list of chunks after embedding (including already-done ones).
    """
    pending = get_unembedded_chunks(session, video_id)

    for chunk in pending:
        vector = embed_text(chunk.content)
        save_embedding(session, chunk, vector)

    # Return all chunks (embedded + already-done) ordered by index
    return list(
        session.exec(
            select(TranscriptChunk)
            .where(TranscriptChunk.video_id == video_id)
            .order_by(TranscriptChunk.chunk_index)
        ).all()
    )
