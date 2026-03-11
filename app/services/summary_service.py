from sqlmodel import Session, select

from app.models.summary import Summary


# ---------------------------------------------------------------------------
# Summary lookup
# ---------------------------------------------------------------------------

def get_summary_by_video_id(session: Session, video_id: int) -> Summary | None:
    """
    Return the summary for a given video, or None if it doesn't exist yet.

    The summaries table has a unique constraint on video_id (one video →
    one summary), so .first() is correct — there can never be more than one.
    """
    return session.exec(
        select(Summary).where(Summary.video_id == video_id)
    ).first()


# ---------------------------------------------------------------------------
# Summary generation
#
# generate_summary() is intentionally isolated so you can swap the
# implementation later without touching the router or the DB logic.
#
# Current implementation: extractive — picks the first N sentences.
# Future swap: replace the body with an OpenAI / Anthropic API call.
# ---------------------------------------------------------------------------

def generate_summary(text: str, max_sentences: int = 5) -> str:
    """
    Produce a summary from raw transcript text.

    Current strategy: return the first `max_sentences` sentences.
    This is fast, offline, and requires no extra dependencies.

    To upgrade to AI-generated summaries later, replace this function body
    with an API call (e.g. OpenAI ChatCompletion) — the rest of the code
    stays the same.

    Args:
        text:           The full transcript string.
        max_sentences:  How many sentences to include in the summary.

    Returns:
        A summary string. Returns the full text if it is shorter than
        max_sentences sentences.
    """
    # Split on common sentence-ending punctuation.
    # This is intentionally simple — good enough for development and testing.
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    selected = sentences[:max_sentences]
    return " ".join(selected)


# ---------------------------------------------------------------------------
# Summary persistence
# ---------------------------------------------------------------------------

def create_summary(
    session: Session,
    video_id: int,
    content: str,
    source: str = "local",
) -> Summary:
    """
    Persist a new Summary row linked to the given video.

    The caller (router) is responsible for the idempotency check — this
    function always inserts and does not check for an existing summary.

    Args:
        session:   The database session injected by FastAPI.
        video_id:  The primary key of the video this summary belongs to.
        content:   The generated summary text.
        source:    How the summary was produced — "local" by default.

    Returns:
        The newly created, database-refreshed Summary object.
    """
    summary = Summary(video_id=video_id, content=content, source=source)
    session.add(summary)
    session.commit()
    session.refresh(summary)
    return summary
