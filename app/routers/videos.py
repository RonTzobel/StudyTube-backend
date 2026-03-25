"""
videos.py — Video management router.

All endpoints require a valid Bearer JWT.
Ownership is enforced on every video-scoped operation using _get_owned_video().

Ownership rules:
  - 401 : missing or invalid token (enforced by get_current_user dependency)
  - 404 : resource does not exist
  - 403 : resource exists but belongs to a different user
"""

import logging
import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, File
from sqlmodel import Session

_log = logging.getLogger(__name__)

from app.config.settings import settings
from app.core.dependencies import get_current_user
from app.database.session import get_session
from app.models.user import User
from app.models.video import Video
from app.schemas.chunk import ChunkRead
from app.schemas.qa import AskRequest, AskResponse
from app.schemas.quiz import QuizRequest, QuizResponse
from app.schemas.retrieval import RetrievedChunkRead, SearchChunksRequest
from app.schemas.summary import SummaryRead
from app.schemas.transcript import TranscriptRead
from app.schemas.video import VideoRead, TranscribeAccepted, UploadResponse
from app.services.s3_service import upload_file_to_s3, delete_file_from_s3
from app.services.summary_service import (
    create_summary,
    generate_summary,
    get_summary_by_video_id,
)
from app.services.transcript_service import (
    get_transcript_by_video_id,
    get_video_by_id,
)
from app.worker.redis_conn import default_queue

# Job function referenced by dotted path so FastAPI never imports app.worker.jobs.
# This keeps the Whisper model out of the API server process entirely.
_PIPELINE_JOB = "app.worker.jobs.process_video_pipeline"
from app.services.chunk_service import (
    chunks_exist_for_video,
    create_chunks,
    get_chunks_by_video_id,
    split_into_chunks,
)
from app.services.embedding_service import embed_all_chunks
from app.services.qa_service import answer_question
from app.services.quiz_service import generate_quiz
from app.services.retrieval_service import search_chunks
from app.services.video_service import (
    create_video_from_upload,
    delete_video,
    get_videos_for_user,
    update_video_status,
)

# Allowed video MIME types — reject anything not in this whitelist.
ALLOWED_CONTENT_TYPES = {
    "video/mp4",
    "video/webm",
    "video/ogg",
    "video/quicktime",  # .mov
    "video/x-msvideo",  # .avi
}

# 500 MB hard limit.
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024

router = APIRouter(prefix="/videos", tags=["Videos"])


# ---------------------------------------------------------------------------
# Private ownership helper
# ---------------------------------------------------------------------------

def _get_owned_video(session: Session, video_id: int, user_id: int) -> Video:
    """
    Fetch a video and verify the requesting user owns it.

    Raises:
        404 if the video does not exist.
        403 if the video exists but belongs to a different user.

    Using 403 (not 404) for the ownership failure case is intentional:
    it tells the client "you are authenticated but this resource is not yours",
    which is more honest than pretending it doesn't exist once the user is
    already logged in. If you prefer to hide existence entirely, swap to 404.
    """
    video = get_video_by_id(session, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found.")
    if video.user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to access this video.",
        )
    return video


_PIPELINE_IN_PROGRESS = {"queued", "processing", "transcribing", "embedding"}


def _require_ready(video: Video) -> None:
    """
    Raise an appropriate HTTP error if the video is not yet fully processed.

    409  — pipeline is still running (queued / processing / transcribing / embedding).
    422  — pipeline failed or video has never been transcribed.
    """
    if video.status in _PIPELINE_IN_PROGRESS:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Video {video.id} is still being processed "
                f"(status='{video.status}'). "
                "Poll GET /api/v1/videos/{video_id} until status is 'completed'."
            ),
        )
    if video.status == "failed":
        raise HTTPException(
            status_code=422,
            detail=(
                f"Video {video.id} processing failed. "
                "Re-upload and transcribe to try again."
            ),
        )
    if video.status != "completed":
        raise HTTPException(
            status_code=422,
            detail=(
                f"Video {video.id} is not ready (status='{video.status}'). "
                "Run POST /transcribe first."
            ),
        )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=UploadResponse, status_code=201)
def upload_video(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Accept a video file, upload it to S3, and create an owned Video metadata row.

    The video binary is stored in S3 — PostgreSQL receives only the S3 key.
    Ownership is derived exclusively from the authenticated JWT.
    """
    # 1 — validate MIME type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{file.content_type}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )

    # 2 — validate file size without buffering the whole file into memory
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum allowed size is 500 MB.",
        )

    # 3 — upload to S3 (stream-upload, no disk write)
    # If this fails, no DB row is created — the error surfaces to the caller.
    try:
        s3_key = upload_file_to_s3(file=file, user_id=current_user.id)
    except Exception as exc:
        _log.error("s3 upload error | user_id=%d | %s", current_user.id, exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to upload the video to storage. Please try again.",
        )

    # 4 — create the metadata row only after S3 confirms success
    original_filename = file.filename or "untitled"
    video = create_video_from_upload(
        session=session,
        title=original_filename,
        user_id=current_user.id,       # ← JWT identity, never from request body
        s3_key=s3_key,
        original_filename=original_filename,
    )

    _log.info(
        "upload done | video_id=%d  user_id=%d  s3_key=%s",
        video.id, current_user.id, s3_key,
    )

    return UploadResponse(
        message="Video uploaded successfully. Call POST /transcribe to start processing.",
        video_id=video.id,
        s3_key=s3_key,
        status=video.status,
    )


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

@router.get("/", response_model=List[VideoRead])
def list_videos(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Return all videos belonging to the authenticated user.

    Only the caller's own videos are returned — no user_id filter from
    the frontend is read or trusted.
    """
    return get_videos_for_user(session, user_id=current_user.id)


# ---------------------------------------------------------------------------
# Get single video (useful for polling transcription status)
# ---------------------------------------------------------------------------

@router.get("/{video_id}", response_model=VideoRead)
def get_video(
    video_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Return a single owned video by ID.

    Intended for status polling after POST /transcribe:
    call this endpoint repeatedly until `status` is "completed" or "failed".
    """
    return _get_owned_video(session, video_id, current_user.id)


@router.get("/{video_id}/status", response_model=VideoRead)
def get_video_status(
    video_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Return the current processing status of an owned video.

    Lightweight polling endpoint — identical response shape to
    GET /{video_id} so the frontend can use either.

    Status values:
      "uploaded"     → file saved, pipeline not yet started
      "queued"       → job accepted by Redis, worker not yet picked it up
      "processing"   → worker extracting audio with ffmpeg
      "transcribing" → worker running Whisper transcription
      "embedding"    → chunking / embedding in progress
      "completed"    → all done; quiz, chat, and search are available
      "failed"       → pipeline error (see error_message field)
    """
    return _get_owned_video(session, video_id, current_user.id)


# ---------------------------------------------------------------------------
# Transcript — read
# ---------------------------------------------------------------------------

@router.get("/{video_id}/transcript", response_model=TranscriptRead)
def get_transcript(
    video_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Return the transcript for a video the caller owns."""
    _get_owned_video(session, video_id, current_user.id)  # ownership gate

    transcript = get_transcript_by_video_id(session, video_id)
    if transcript is None:
        raise HTTPException(
            status_code=404,
            detail=f"No transcript found for video {video_id}. "
                   "Run POST /transcribe first.",
        )
    return transcript


# ---------------------------------------------------------------------------
# Transcribe
# ---------------------------------------------------------------------------

@router.post("/{video_id}/transcribe", response_model=TranscribeAccepted)
def transcribe_video(
    video_id: int,
    response: Response,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Enqueue the full video processing pipeline via Redis/RQ.

    Returns 202 immediately — a worker process picks up the job and runs:
      ffmpeg extraction → Whisper transcription → chunking → embedding.

    Poll GET /api/v1/videos/{video_id} and check `status`:
      "queued"       → job accepted, waiting for a free worker
      "processing"   → worker extracting audio with ffmpeg
      "transcribing" → worker running Whisper transcription
      "embedding"    → chunking / embedding in progress
      "completed"    → all done; all endpoints available
      "failed"       → pipeline error (see error_message field)

    Returns 200 if the video is already ready (no new job is started).
    Returns 409 if the pipeline is already queued or running.
    """
    video = _get_owned_video(session, video_id, current_user.id)

    has_file = bool(video.s3_key) or (
        bool(video.file_path) and Path(video.file_path).exists()
    )
    if not has_file:
        raise HTTPException(
            status_code=422,
            detail=f"Video {video_id} has no accessible file. Upload the file first.",
        )

    # Already fully processed — nothing to do.
    if video.status == "completed":
        response.status_code = 200
        return TranscribeAccepted(
            message="Video is already processed. Fetch the transcript at GET /transcript.",
            video_id=video_id,
            status="completed",
        )

    # Already queued or running — reject to avoid duplicate jobs.
    if video.status in _PIPELINE_IN_PROGRESS:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Video {video_id} is already being processed "
                f"(status='{video.status}'). "
                "Poll GET /api/v1/videos/{video_id} for status updates."
            ),
        )

    # Enqueue FIRST — if Redis is unavailable, do NOT leave the video stuck
    # in "queued" with no real job behind it (that would block all future
    # retry attempts since "queued" is in _PIPELINE_IN_PROGRESS).
    try:
        default_queue.enqueue(
            _PIPELINE_JOB,
            video_id,
            job_timeout=settings.VIDEO_PIPELINE_JOB_TIMEOUT,
        )
    except Exception as enqueue_exc:
        _log.error(
            "enqueue failed | video_id=%d | %s: %s",
            video_id, type(enqueue_exc).__name__, enqueue_exc,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Failed to queue the processing job. "
                "Redis may be unavailable — try again shortly."
            ),
        )

    # Job is confirmed in the queue — now persist the status change.
    update_video_status(session, video, "queued")

    _log.info("job enqueued | video_id=%d | timeout=%ds", video_id, settings.VIDEO_PIPELINE_JOB_TIMEOUT)

    response.status_code = 202
    return TranscribeAccepted(
        message=(
            "Job queued. "
            "Poll GET /api/v1/videos/{video_id} until status is 'ready', "
            "then fetch the transcript at GET /api/v1/videos/{video_id}/transcript."
        ),
        video_id=video_id,
        status="queued",
    )


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------

@router.post("/{video_id}/summarize", response_model=SummaryRead)
def summarize_video(
    video_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Generate and save a summary for an owned video's transcript."""
    _get_owned_video(session, video_id, current_user.id)  # ownership gate

    transcript = get_transcript_by_video_id(session, video_id)
    if transcript is None:
        raise HTTPException(
            status_code=422,
            detail=f"No transcript for video {video_id}. Run /transcribe first.",
        )

    existing = get_summary_by_video_id(session, video_id)
    if existing is not None:
        return existing  # idempotent

    summary_text = generate_summary(transcript.content or "")
    return create_summary(session, video_id=video_id, content=summary_text, source="local")


# ---------------------------------------------------------------------------
# Chunk — create
# ---------------------------------------------------------------------------

@router.post("/{video_id}/chunk", response_model=List[ChunkRead])
def chunk_transcript(
    video_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Split an owned video's transcript into chunks."""
    _get_owned_video(session, video_id, current_user.id)  # ownership gate

    transcript = get_transcript_by_video_id(session, video_id)
    if transcript is None:
        raise HTTPException(
            status_code=422,
            detail=f"No transcript for video {video_id}. Run /transcribe first.",
        )

    if chunks_exist_for_video(session, video_id):
        return get_chunks_by_video_id(session, video_id)  # idempotent

    raw_chunks = split_into_chunks(transcript.content or "")
    return create_chunks(session, video_id=video_id, chunks=raw_chunks)


# ---------------------------------------------------------------------------
# Chunk — list
# ---------------------------------------------------------------------------

@router.get("/{video_id}/chunks", response_model=List[ChunkRead])
def list_chunks(
    video_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Return all chunks for an owned video."""
    _get_owned_video(session, video_id, current_user.id)  # ownership gate
    return get_chunks_by_video_id(session, video_id)


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------

@router.post("/{video_id}/embed", response_model=List[ChunkRead])
def embed_video_chunks(
    video_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Generate embeddings for all chunks of an owned video."""
    _get_owned_video(session, video_id, current_user.id)  # ownership gate

    if not chunks_exist_for_video(session, video_id):
        raise HTTPException(
            status_code=422,
            detail=f"No chunks for video {video_id}. Run /chunk first.",
        )
    return embed_all_chunks(session, video_id)


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

@router.post("/{video_id}/search", response_model=List[RetrievedChunkRead])
def search_video_chunks(
    video_id: int,
    request: SearchChunksRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Semantic search over an owned video's transcript chunks."""
    video = _get_owned_video(session, video_id, current_user.id)
    _require_ready(video)

    try:
        return search_chunks(
            session=session,
            video_id=video_id,
            query=request.query,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ---------------------------------------------------------------------------
# Ask (RAG)
# ---------------------------------------------------------------------------

@router.post("/{video_id}/ask", response_model=AskResponse)
def ask_question(
    video_id: int,
    request: AskRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Answer a question about an owned video using RAG."""
    video = _get_owned_video(session, video_id, current_user.id)
    _require_ready(video)

    try:
        return answer_question(
            session=session,
            video_id=video_id,
            question=request.question,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Quiz
# ---------------------------------------------------------------------------

@router.post("/{video_id}/quiz", response_model=QuizResponse)
def generate_video_quiz(
    video_id: int,
    request: QuizRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Generate a quiz grounded in an owned video's transcript."""
    video = _get_owned_video(session, video_id, current_user.id)
    _require_ready(video)

    try:
        return generate_quiz(
            session=session,
            video_id=video_id,
            num_questions=request.num_questions,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

@router.delete("/{video_id}", status_code=204)
def delete_video_endpoint(
    video_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Permanently delete a video owned by the authenticated user.

    Ownership is enforced via _get_owned_video():
      - 401 if the JWT is missing or invalid
      - 404 if the video does not exist
      - 403 if the video belongs to a different user
    """
    video = _get_owned_video(session, video_id, current_user.id)

    # Remove the video file from wherever it lives.
    # S3 (new uploads) and local disk (legacy) are both handled.
    if video.s3_key:
        delete_file_from_s3(video.s3_key)
    elif video.file_path:
        file = Path(video.file_path)
        if file.exists():
            file.unlink()

    delete_video(session, video)
    return Response(status_code=204)
