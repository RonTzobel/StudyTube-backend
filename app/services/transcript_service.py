import logging

from sqlmodel import Session, select

from app.models.transcript import Transcript
from app.models.video import Video

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Video / transcript lookups
# ---------------------------------------------------------------------------

def get_video_by_id(session: Session, video_id: int) -> Video | None:
    """
    Look up a video by its primary key.

    Returning None (instead of raising) keeps the decision about HTTP status
    codes in the router where it belongs.
    """
    return session.get(Video, video_id)


def get_transcript_by_video_id(session: Session, video_id: int) -> Transcript | None:
    """
    Return the transcript for a given video, or None if it doesn't exist yet.

    The transcripts table has a unique constraint on video_id (one video →
    one transcript), so .first() is the correct call here.
    """
    return session.exec(
        select(Transcript).where(Transcript.video_id == video_id)
    ).first()


# ---------------------------------------------------------------------------
# Transcript persistence
# ---------------------------------------------------------------------------

def create_transcript(
    session: Session,
    video_id: int,
    content: str,
    source: str = "whisper",
) -> Transcript:
    """
    Persist a new Transcript row linked to the given video.

    The caller is responsible for the idempotency check so the error
    message and HTTP status code stay in the router layer.
    """
    transcript = Transcript(video_id=video_id, content=content, source=source)
    session.add(transcript)
    session.commit()
    session.refresh(transcript)
    return transcript
