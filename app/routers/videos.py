import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlmodel import Session

from app.config.settings import settings
from app.database.session import get_session
from app.schemas.transcript import TranscriptRead
from app.schemas.video import VideoRead
from app.services.transcript_service import (
    create_transcript,
    extract_audio,
    get_transcript_by_video_id,
    get_video_by_id,
    transcribe_audio,
)
from app.services.video_service import create_video_from_upload, save_upload_file, update_video_status

# Allowed video MIME types.
# This is a basic whitelist — we reject anything that doesn't look like a video.
ALLOWED_CONTENT_TYPES = {
    "video/mp4",
    "video/webm",
    "video/ogg",
    "video/quicktime",  # .mov
    "video/x-msvideo",  # .avi
}

# 500 MB limit — prevents the server from being overwhelmed by huge files.
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024

router = APIRouter(prefix="/videos", tags=["Videos"])


@router.post("/upload", response_model=VideoRead)
def upload_video(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    """
    Accept a video file, save it to disk, and create a Video record in the DB.

    How it works:
      1. Validate the file's MIME type (must be a video format we support).
      2. Validate the file size (must not exceed MAX_FILE_SIZE_BYTES).
      3. Save the file to the uploads folder (via the service layer).
      4. Create a Video record in the database (via the service layer).
      5. Return the saved Video record as JSON.

    The 'session' parameter is injected automatically by FastAPI using
    Depends(get_session). The router never creates a session directly.

    The 'response_model=VideoRead' tells FastAPI to serialize the returned
    Video object using the VideoRead schema — this also drives the Swagger docs.
    """
    # Step 1 — validate the MIME type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{file.content_type}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )

    # Step 2 — validate the file size using seek/tell (no memory cost)
    file.file.seek(0, 2)          # jump to end
    file_size = file.file.tell()  # position at end = total bytes
    file.file.seek(0)             # rewind before saving

    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum allowed size is 500 MB.",
        )

    # Step 3 — save the file to disk
    result = save_upload_file(
        upload_file=file,
        upload_dir=settings.UPLOAD_DIR,
    )

    # Step 4 — create a Video record in the database
    # The title defaults to the original filename until the user can edit it.
    # user_id=1 is hardcoded — will come from JWT token once auth is implemented.
    video = create_video_from_upload(
        session=session,
        title=file.filename or "untitled",
        file_path=result["file_path"],
    )

    # Step 5 — return the saved Video object
    # FastAPI will serialize it using the VideoRead schema (see response_model above).
    return video


@router.get("/{video_id}/transcript", response_model=TranscriptRead)
def get_transcript(
    video_id: int,
    session: Session = Depends(get_session),
):
    """
    Return the transcript for a given video.

    Raises 404 if the video does not exist.
    Raises 404 if the video exists but has no transcript yet.

    The two checks are intentionally separate so the error message tells
    the client exactly what is missing.
    """
    # Step 1 — confirm the video exists
    # We check this first so the client gets a clear "video not found" message
    # rather than a confusing "transcript not found" for a non-existent video.
    video = get_video_by_id(session, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found.")

    # Step 2 — fetch the transcript
    transcript = get_transcript_by_video_id(session, video_id)
    if transcript is None:
        raise HTTPException(
            status_code=404,
            detail=f"No transcript found for video {video_id}. "
                   "Transcription has not been run yet.",
        )

    # Step 3 — return the transcript
    # FastAPI serializes it using TranscriptRead (see response_model above).
    return transcript


@router.post("/{video_id}/transcribe", response_model=TranscriptRead)
def transcribe_video(
    video_id: int,
    session: Session = Depends(get_session),
):
    """
    Trigger Whisper transcription for an already-uploaded video.

    How it works:
      1. Confirm the video exists and its file is present on disk.
      2. Return the existing transcript immediately if one already exists
         (idempotent — safe to call more than once).
      3. Mark the video as "processing" so callers know work is in progress.
      4. Extract audio from the video file using ffmpeg (temp WAV, 16 kHz mono).
      5. Transcribe the audio with OpenAI Whisper.
      6. Save the transcript to the DB and mark the video "done".
      7. Clean up the temp audio file regardless of success or failure.

    This endpoint is synchronous — the request blocks until transcription
    finishes. For long videos this can take minutes on CPU. A background
    task queue (Celery / ARQ) is the right fix, but is out of scope for now.

    Prerequisites (must be installed before calling this endpoint):
      - ffmpeg must be on your PATH  (see README for install instructions)
      - pip install openai-whisper   (or it's in requirements.txt)
    """
    # Step 1 — confirm the video exists and has a file on disk
    video = get_video_by_id(session, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found.")

    if not video.file_path or not Path(video.file_path).exists():
        raise HTTPException(
            status_code=422,
            detail=(
                f"Video {video_id} has no file on disk. "
                "Upload the file first via POST /api/v1/videos/upload."
            ),
        )

    # Step 2 — idempotency: return existing transcript without re-running Whisper
    existing = get_transcript_by_video_id(session, video_id)
    if existing is not None:
        return existing

    # Step 3 — mark as processing before the slow work starts
    update_video_status(session, video, "processing")

    # Steps 4 & 5 — extract audio then transcribe
    # We keep audio_path in a variable so the finally block can always clean up.
    audio_path = None
    try:
        audio_path = extract_audio(video.file_path)
        text = transcribe_audio(audio_path)
    except FileNotFoundError:
        update_video_status(session, video, "failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "ffmpeg was not found. "
                "Install ffmpeg and make sure it is on your system PATH."
            ),
        )
    except RuntimeError as exc:
        update_video_status(session, video, "failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # Always delete the temp WAV — even if transcription raised an exception
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

    # Steps 6 & 7 — persist transcript, update status
    transcript = create_transcript(session, video_id=video_id, content=text, source="whisper")
    update_video_status(session, video, "done")
    return transcript


@router.get("/")
def list_videos_placeholder():
    """Placeholder — will list the current user's videos once auth is in place."""
    return {"message": "Videos router is ready. Implementation coming soon."}
