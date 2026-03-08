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


def delete_video(session: Session, video: Video) -> None:
    """Delete a video record from the database."""
    session.delete(video)
    session.commit()
