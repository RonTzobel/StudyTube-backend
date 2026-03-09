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
    user_id: int = 1,
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
        user_id:   Hardcoded to 1 until JWT auth is implemented. Every upload
                   is owned by the placeholder user created at startup.

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
    """Delete a video record from the database."""
    session.delete(video)
    session.commit()
