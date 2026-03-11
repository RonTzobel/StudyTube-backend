from typing import List

from sqlmodel import Session, select

from app.models.chunk import TranscriptChunk


# ---------------------------------------------------------------------------
# Chunking logic
#
# split_into_chunks() is intentionally isolated so the strategy can be
# swapped later (e.g. sentence-aware splitting, tiktoken-based splitting)
# without touching anything else.
# ---------------------------------------------------------------------------

def split_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[dict]:
    """
    Split a transcript into fixed-size character chunks with overlap.

    Why overlap?
    A sentence that falls on a chunk boundary would be cut in half.
    Overlap ensures the beginning of each chunk repeats the end of the
    previous one, so no context is lost at boundaries.

    Args:
        text:        The full transcript string.
        chunk_size:  Maximum number of characters per chunk.
        overlap:     How many characters to repeat from the previous chunk.

    Returns:
        A list of dicts, each with keys:
            chunk_index  — 0-based position in the sequence
            content      — the chunk text
            start_char   — starting character offset in the original text
            end_char     — ending character offset in the original text
    """
    text = text.strip()
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        chunks.append({
            "chunk_index": index,
            "content": chunk_text,
            "start_char": start,
            "end_char": end,
        })

        # Advance by chunk_size minus overlap so consecutive chunks share context
        start += chunk_size - overlap
        index += 1

    return chunks


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_chunks_by_video_id(session: Session, video_id: int) -> List[TranscriptChunk]:
    """
    Return all chunks for a given video, ordered by chunk_index.

    Returns an empty list if no chunks exist yet (caller decides what HTTP
    status that maps to).
    """
    return list(
        session.exec(
            select(TranscriptChunk)
            .where(TranscriptChunk.video_id == video_id)
            .order_by(TranscriptChunk.chunk_index)
        ).all()
    )


def chunks_exist_for_video(session: Session, video_id: int) -> bool:
    """Return True if at least one chunk already exists for this video."""
    result = session.exec(
        select(TranscriptChunk).where(TranscriptChunk.video_id == video_id).limit(1)
    ).first()
    return result is not None


def create_chunks(
    session: Session,
    video_id: int,
    chunks: List[dict],
) -> List[TranscriptChunk]:
    """
    Bulk-insert a list of chunk dicts produced by split_into_chunks().

    All chunks are added in a single commit for efficiency.

    Args:
        session:   The database session injected by FastAPI.
        video_id:  The primary key of the video these chunks belong to.
        chunks:    List of dicts from split_into_chunks().

    Returns:
        The list of persisted TranscriptChunk objects.
    """
    db_chunks = [
        TranscriptChunk(
            video_id=video_id,
            chunk_index=c["chunk_index"],
            content=c["content"],
            start_char=c["start_char"],
            end_char=c["end_char"],
        )
        for c in chunks
    ]

    for chunk in db_chunks:
        session.add(chunk)

    session.commit()

    for chunk in db_chunks:
        session.refresh(chunk)

    return db_chunks
