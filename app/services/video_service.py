import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlmodel import Session, select

from app.models.video import Video
from app.schemas.video import VideoCreate


def save_upload_file(upload_file: UploadFile, upload_dir: str) -> dict:
    """
    Save an uploaded video file to the local uploads folder.

    Steps:
      1. Make sure the uploads directory exists (create it if not).
      2. Generate a unique filename so two uploads with the same original
         name never overwrite each other.
      3. Write the file content to disk in chunks — this avoids loading
         the entire file into memory at once (important for large videos).
      4. Return the saved filename and its full path.

    Args:
        upload_file: The file object provided by FastAPI from the request.
        upload_dir:  Path to the directory where files should be saved.

    Returns:
        A dict with 'saved_filename' and 'file_path'.
    """
    # Step 1 — ensure the uploads directory exists
    destination_dir = Path(upload_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Step 2 — build a unique filename
    # We prefix the original name with a UUID so collisions are impossible.
    # Example: "lecture.mp4" → "3f2a1b..._lecture.mp4"
    original_name = upload_file.filename or "video"
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    file_path = destination_dir / unique_name

    # Step 3 — write to disk in 1 MB chunks
    chunk_size = 1024 * 1024  # 1 MB
    with open(file_path, "wb") as buffer:
        while chunk := upload_file.file.read(chunk_size):
            buffer.write(chunk)

    # Step 4 — return info the router can pass back to the client
    return {
        "saved_filename": unique_name,
        "file_path": str(file_path),
    }


def create_video_from_upload(
    session: Session,
    title: str,
    file_path: str,
    user_id: int,
) -> Video:
    """
    Create a Video record in the database after a file has been saved to disk.

    This is called by the upload endpoint once save_upload_file() succeeds.
    It is intentionally separate from create_video() because the upload flow
    receives a filename and a file path — not a VideoCreate schema.

    Args:
        session:   The database session injected by FastAPI.
        title:     The video title — defaults to the original filename for now.
        file_path: The path where the file was saved on disk.
        user_id:   The authenticated user's ID from the JWT token.
                   Must be provided by the router — never a default.

    Returns:
        The newly created and database-refreshed Video object.
    """
    video = Video(
        user_id=user_id,
        title=title,
        file_path=file_path,
        status="uploaded",
    )
    # session.add() stages the object — it is not in the DB yet.
    session.add(video)
    # session.commit() writes the INSERT to PostgreSQL and assigns the id.
    session.commit()
    # session.refresh() re-reads the row so the returned object has
    # the id and created_at values that PostgreSQL generated.
    session.refresh(video)
    return video


def get_videos_for_user(session: Session, user_id: int) -> list[Video]:
    """Return all videos belonging to a specific user."""
    return list(session.exec(select(Video).where(Video.user_id == user_id)).all())


def get_video_by_id(session: Session, video_id: int) -> Video | None:
    """Look up a single video by its primary key."""
    return session.get(Video, video_id)


def create_video(session: Session, user_id: int, video_data: VideoCreate) -> Video:
    """
    Register a new video record in the database.

    File upload/storage handling will be added in a future step.
    For now we only persist the metadata.
    """
    video = Video(
        user_id=user_id,
        title=video_data.title,
        description=video_data.description,
        status="pending",
    )
    session.add(video)
    session.commit()
    session.refresh(video)
    return video


def update_video_status(session: Session, video: Video, status: str) -> None:
    """
    Update the processing status of a video and commit immediately.

    Valid status values (by convention): uploaded → processing → done → failed

    This is called by the transcription endpoint to keep the video record
    in sync with what is actually happening to the file.
    """
    video.status = status
    session.add(video)
    session.commit()
    session.refresh(video)


def delete_video(session: Session, video: Video) -> None:
    """
    Delete a video and all child records that reference it.

    Deletion order (child → parent to satisfy FK constraints):
      1. chat_messages  (FK → chat_sessions.id)
      2. chat_sessions  (FK → videos.id)
      3. transcript_chunks (FK → videos.id)
      4. transcripts    (FK → videos.id)
      5. summaries      (FK → videos.id)
      6. video          (parent)
    """
    import logging
    from app.models.chat import ChatSession, ChatMessage
    from app.models.chunk import TranscriptChunk
    from app.models.transcript import Transcript
    from app.models.summary import Summary

    log = logging.getLogger(__name__)
    log.info("delete_video: starting deletion for video_id=%s", video.id)

    # 1. Chat messages (must go before sessions)
    chat_sessions = session.exec(
        select(ChatSession).where(ChatSession.video_id == video.id)
    ).all()
    for cs in chat_sessions:
        messages = session.exec(
            select(ChatMessage).where(ChatMessage.session_id == cs.id)
        ).all()
        for msg in messages:
            session.delete(msg)
    session.flush()

    # 2. Chat sessions
    for cs in chat_sessions:
        session.delete(cs)
    session.flush()

    # 3. Transcript chunks
    chunks = session.exec(
        select(TranscriptChunk).where(TranscriptChunk.video_id == video.id)
    ).all()
    for chunk in chunks:
        session.delete(chunk)
    session.flush()

    # 4. Transcript
    transcript = session.exec(
        select(Transcript).where(Transcript.video_id == video.id)
    ).first()
    if transcript:
        session.delete(transcript)
        session.flush()

    # 5. Summary
    summary = session.exec(
        select(Summary).where(Summary.video_id == video.id)
    ).first()
    if summary:
        session.delete(summary)
        session.flush()

    # 6. Video (parent)
    session.delete(video)
    session.commit()
    log.info("delete_video: successfully deleted video_id=%s", video.id)
