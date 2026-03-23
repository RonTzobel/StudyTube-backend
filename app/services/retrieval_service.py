import json
import math
from typing import List

from sqlmodel import Session, select

from app.models.chunk import TranscriptChunk
from app.schemas.retrieval import RetrievedChunkRead
from app.services.embedding_service import embed_text


# ---------------------------------------------------------------------------
# Cosine similarity
#
# Cosine similarity measures the angle between two vectors, not their size.
# A score of 1.0 means the vectors point in the exact same direction
# (maximum semantic similarity). A score of 0.0 means they are orthogonal
# (unrelated). A score of -1.0 means opposite meaning.
#
# For sentence embeddings the practical range is roughly 0.0 – 1.0 because
# the model is trained to produce non-negative similarity for related text.
#
# Formula:
#   cosine_similarity(A, B) = (A · B) / (|A| × |B|)
#
# We implement this in pure Python so no extra dependency is needed.
# numpy would be faster but vectors are small (e.g. 384 floats) — pure Python
# is fast enough here and keeps the dependency list short.
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two equal-length float vectors.

    Returns 0.0 if either vector has zero magnitude (all-zero vector),
    which would normally cause a division-by-zero error.
    """
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_embedded_chunks(session: Session, video_id: int) -> List[TranscriptChunk]:
    """
    Return all chunks for a video that have been embedded (is_embedded=True).

    Only embedded chunks can be searched — chunks without a vector are skipped
    because there is nothing to compare the query against.
    """
    return list(
        session.exec(
            select(TranscriptChunk)
            .where(TranscriptChunk.video_id == video_id)
            .where(TranscriptChunk.is_embedded == True)  # noqa: E712
            .order_by(TranscriptChunk.chunk_index)
        ).all()
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def search_chunks(
    session: Session,
    video_id: int,
    query: str,
    top_k: int = 5,
) -> List[RetrievedChunkRead]:
    """
    Find the most semantically relevant chunks for a query string.

    Steps:
      1. Load all embedded chunks for the video from the DB.
      2. Embed the query using the same model used for chunks.
      3. Compute cosine similarity between the query vector and each chunk.
      4. Sort by similarity descending and return the top_k results.

    Args:
        session:   The database session injected by FastAPI.
        video_id:  Only search chunks belonging to this video.
        query:     The natural-language question or phrase.
        top_k:     How many results to return.

    Returns:
        A list of RetrievedChunkRead objects ordered by similarity (highest first).

    Raises:
        RuntimeError: If sentence-transformers is not installed (from embed_text).
        ValueError:   If no embedded chunks exist for this video.
    """
    # Step 1 — load embedded chunks
    chunks = get_embedded_chunks(session, video_id)
    if not chunks:
        raise ValueError(
            f"No embedded chunks found for video {video_id}. "
            "Transcribe the video first (POST /transcribe) and wait for status 'ready'."
        )

    # Step 2 — embed the query with the same model used for chunks
    query_vector = embed_text(query)

    # Step 3 — score every chunk
    scored = []
    for chunk in chunks:
        if chunk.embedding is None:
            continue  # safety guard; is_embedded=True should guarantee this isn't None
        chunk_vector = json.loads(chunk.embedding)
        score = _cosine_similarity(query_vector, chunk_vector)
        scored.append((score, chunk))

    # Step 4 — sort highest similarity first, take top_k
    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = scored[:top_k]

    return [
        RetrievedChunkRead(
            id=chunk.id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            similarity_score=round(score, 6),
            start_char=chunk.start_char,
            end_char=chunk.end_char,
        )
        for score, chunk in top
    ]
