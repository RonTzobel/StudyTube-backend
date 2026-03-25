import json
from typing import List

from sqlmodel import Session, select

from app.config.settings import settings
from app.models.chunk import TranscriptChunk


# ---------------------------------------------------------------------------
# Embedding model — loaded once at module import time.
#
# Model is configured via EMBEDDING_MODEL_NAME in settings / .env.
# Current default: BAAI/bge-small-en-v1.5
#   - 384-dimensional vectors
#   - Strong English semantic retrieval; MTEB top performer at this size
#   - ~130 MB on disk; fast on CPU
#   - RAG thresholds in settings.py are calibrated for this model
#
# IMPORTANT — model compatibility:
#   Chunk embeddings and query embeddings MUST use the same model.
#   If you change EMBEDDING_MODEL_NAME, re-process every existing video
#   (POST /api/v1/videos/{id}/transcribe) so old chunks are re-embedded.
#   Mixing vectors from different models silently breaks cosine similarity.
#
# To swap to a different model: change EMBEDDING_MODEL_NAME in .env and
# adjust RAG_LOW_THRESHOLD / RAG_GOOD_THRESHOLD in .env to match the new
# model's typical score range.
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer as _ST
    _model = _ST(settings.EMBEDDING_MODEL_NAME)
    _model_name = settings.EMBEDDING_MODEL_NAME
except Exception:
    # sentence-transformers not installed or model failed to load.
    # The server still starts; embed_text() will raise a clear RuntimeError.
    _model = None
    _model_name = None


def embed_text(text: str) -> List[float]:
    """
    Generate a semantic embedding vector for a piece of text.

    Uses the model configured in settings.EMBEDDING_MODEL_NAME.
    No API key or network call is needed after the first run (model is cached
    in ~/.cache/torch/sentence_transformers).

    Args:
        text: The text to embed (chunk content or user query).

    Returns:
        A list of floats representing the embedding vector.
        Length is determined by the configured model (384 for the default).

    Raises:
        RuntimeError: If sentence-transformers is not installed or the model
                      failed to load at startup.
    """
    if _model is None:
        raise RuntimeError(
            f"Embedding model '{settings.EMBEDDING_MODEL_NAME}' is not available. "
            "Check that sentence-transformers is installed and the model name is correct."
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
