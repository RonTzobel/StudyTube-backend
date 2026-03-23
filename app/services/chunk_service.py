import logging
import re
from typing import List

from sqlmodel import Session, select

from app.models.chunk import TranscriptChunk

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transcript normalization
# ---------------------------------------------------------------------------

def _normalize_transcript(text: str) -> str:
    """
    Minimal pre-chunking normalization of Whisper output.

    Whisper can produce:
      - multiple consecutive spaces
      - line breaks in the middle of sentences
      - leading / trailing whitespace

    We only collapse whitespace here — no aggressive stripping that could
    remove Hebrew characters, numbers, or punctuation that matters for
    sentence detection.
    """
    return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# Sentence unit splitter
# ---------------------------------------------------------------------------

# Split after a sentence-ending punctuation mark followed by whitespace.
# The look-behind keeps the punctuation attached to the preceding unit.
# Handles both Hebrew and Latin punctuation (. ! ?) identically.
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')

# Maximum words in a single unit before force-splitting at word boundaries.
# This is the graceful-degradation path for Whisper output that lacks
# sentence punctuation (common in Hebrew lecture transcripts).
_MAX_UNIT_WORDS = 80


def _split_into_sentence_units(text: str) -> List[str]:
    """
    Split text into sentence-like units on terminal punctuation (.!?).

    If a unit produced by punctuation-splitting is longer than
    _MAX_UNIT_WORDS, it is further split at word boundaries so that
    poor punctuation from Whisper degrades gracefully rather than
    producing one giant unsplit blob.

    Returns a list of non-empty strings.
    """
    raw_units = _SENTENCE_BOUNDARY.split(text)

    units: List[str] = []
    for raw in raw_units:
        raw = raw.strip()
        if not raw:
            continue
        words = raw.split()
        if len(words) <= _MAX_UNIT_WORDS:
            units.append(raw)
        else:
            # Long unpunctuated run — force-split at word boundaries
            for i in range(0, len(words), _MAX_UNIT_WORDS):
                part = ' '.join(words[i: i + _MAX_UNIT_WORDS])
                if part:
                    units.append(part)

    return units


# ---------------------------------------------------------------------------
# Chunking logic
#
# split_into_chunks() is the public interface consumed by the pipeline.
# The implementation can be swapped without touching anything else.
# ---------------------------------------------------------------------------

def split_into_chunks(
    text: str,
    target_words: int = 400,
    overlap_words: int = 75,
) -> List[dict]:
    """
    Split a transcript into sentence-aware overlapping chunks.

    Strategy (sentence-aware, word-count targeted):

      1. Normalize whitespace in the transcript.
      2. Split into sentence-like units on terminal punctuation (.!?).
         Units longer than _MAX_UNIT_WORDS are force-split at word
         boundaries so that poor Whisper punctuation degrades gracefully.
      3. Precompute each unit's character offset in the normalized text.
      4. Greedily pack units into a chunk until target_words is reached.
         A unit that would alone exceed target_words is still accepted as a
         single-unit chunk so we never skip content.
      5. Begin the next chunk at the unit that sits overlap_words words
         before the end of the current chunk, so neighbouring chunks share
         context across boundaries without mid-sentence cuts.

    Why word-count, not character-count?
      Hebrew words average ~5 chars; English ~5.5. Word-count gives
      consistent semantic density across both languages. The old 500-char
      limit was ~75–100 words — far too small for coherent lecture chunks.

    Target parameters (calibrated for lecture Q&A):
      target_words  = 400  →  ~2 000–2 500 chars; holds one full idea
      overlap_words =  75  →  ~400 chars; keeps boundary context in both chunks

    Args:
        text:          The full transcript string.
        target_words:  Target word count per chunk.
        overlap_words: Words of overlap between consecutive chunks.

    Returns:
        A list of dicts, each with keys:
            chunk_index  — 0-based position in the sequence
            content      — the chunk text (sentence-aligned)
            start_char   — start offset in the normalized transcript
            end_char     — end offset in the normalized transcript
    """
    norm_text = _normalize_transcript(text)
    units = _split_into_sentence_units(norm_text)

    if not units:
        _log.warning("split_into_chunks | empty transcript — returning no chunks")
        return []

    # Precompute word count and character offset for each unit.
    # We walk through norm_text in order, searching for each unit starting
    # from where the previous one ended. This handles repeated short units
    # (e.g. "כן.") correctly by always advancing the search cursor.
    unit_wc: List[int] = []
    unit_start_char: List[int] = []
    search_pos = 0
    for unit in units:
        pos = norm_text.find(unit, search_pos)
        if pos == -1:
            pos = search_pos  # safety fallback — should not happen
        unit_start_char.append(pos)
        unit_wc.append(len(unit.split()))
        search_pos = pos + len(unit)

    # Build chunks using a greedy unit-packing loop with overlap rewind.
    chunks: List[dict] = []
    chunk_index = 0
    start_unit = 0
    n = len(units)

    while start_unit < n:
        # --- fill this chunk with units until target_words is reached ---
        end_unit = start_unit
        wc_so_far = 0

        while end_unit < n:
            candidate_wc = unit_wc[end_unit]
            # Stop if adding this unit would exceed the target AND we
            # already have some content. The "wc_so_far > 0" guard ensures
            # we always take at least one unit even if it alone exceeds target.
            if wc_so_far + candidate_wc > target_words and wc_so_far > 0:
                break
            wc_so_far += candidate_wc
            end_unit += 1

        # Safety: always advance at least one unit to prevent infinite loop
        if end_unit == start_unit:
            end_unit = start_unit + 1

        chunk_text = ' '.join(units[start_unit:end_unit])
        start_char = unit_start_char[start_unit]
        end_char = unit_start_char[end_unit - 1] + len(units[end_unit - 1])

        chunks.append({
            "chunk_index": chunk_index,
            "content":     chunk_text,
            "start_char":  start_char,
            "end_char":    end_char,
        })
        chunk_index += 1

        # If we just consumed the last unit there is nothing left to chunk.
        # Stop here — the overlap rewind below would otherwise create tiny
        # duplicate tail chunks for the remaining overlap window.
        if end_unit == n:
            break

        # --- compute next chunk's start using overlap rewind ---
        # Walk backwards from end_unit, accumulating word counts, until we
        # have covered overlap_words. That unit is where the next chunk starts.
        overlap_acc = 0
        next_start = end_unit
        while next_start > start_unit + 1:
            next_start -= 1
            overlap_acc += unit_wc[next_start]
            if overlap_acc >= overlap_words:
                break

        # Guarantee forward progress: next chunk must start after start_unit
        start_unit = max(start_unit + 1, next_start)

    # Structured log for pipeline monitoring
    total_words = sum(unit_wc)
    avg_words = total_words / len(chunks) if chunks else 0
    _log.info(
        "split_into_chunks | units=%d  chunks=%d  "
        "avg_words=%.0f  target=%d  overlap=%d  total_words=%d",
        len(units), len(chunks), avg_words, target_words, overlap_words, total_words,
    )

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


def delete_chunks_for_video(session: Session, video_id: int) -> int:
    """
    Delete all existing chunks (and their embeddings) for a video.

    Returns the number of rows deleted.

    Called by the pipeline before re-creating chunks so that a retry
    after a failed "indexing" stage does not produce duplicate rows.
    Embeddings are stored as columns on TranscriptChunk, so deleting
    the chunks implicitly clears the embeddings too.
    """
    chunks = get_chunks_by_video_id(session, video_id)
    count = len(chunks)
    for chunk in chunks:
        session.delete(chunk)
    if count:
        session.commit()
    return count


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
