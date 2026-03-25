"""
jobs.py — RQ job functions executed by the worker process.

ALL heavy computation lives here:
  - ffmpeg audio extraction
  - faster-whisper transcription
  - transcript chunking
  - sentence-transformers embedding

FastAPI NEVER imports this module, so the Whisper model and the embedding
model are loaded only in the worker process, not in the API server.

Each job function opens its own database session via the shared SQLAlchemy
engine — it does NOT use FastAPI request-scoped dependencies.

Pipeline status transitions:
  queued → processing → transcribing → embedding → completed
  queued → processing → failed  (on any unhandled exception)
"""

import logging
import os
import subprocess
import tempfile
import time
from typing import List

from sqlmodel import Session

from app.config.settings import settings
from app.database.session import engine
import app.models  # noqa: F401 — registers ALL models with SQLAlchemy metadata
from app.models.video import Video
from app.services.chunk_service import create_chunks, delete_chunks_for_video, split_into_chunks
from app.services.embedding_service import embed_all_chunks
from app.services.s3_service import download_file_from_s3
from app.services.transcript_service import (
    create_transcript,
    get_transcript_by_video_id,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Whisper model — loaded once when the worker imports this module.
#
# Loading at import time means:
#   1. The model is ready before the first job arrives (no cold-start latency).
#   2. If the model file is missing or broken, the worker fails at startup
#      rather than silently failing mid-job.
# ---------------------------------------------------------------------------

try:
    from faster_whisper import WhisperModel as _WhisperModel

    _whisper_model = _WhisperModel(
        settings.WHISPER_MODEL,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
        cpu_threads=settings.WHISPER_CPU_THREADS,
    )
    _log.info(
        "faster-whisper loaded | model=%s device=%s compute=%s "
        "beam_size=%s vad=%s cpu_threads=%s",
        settings.WHISPER_MODEL,
        settings.WHISPER_DEVICE,
        settings.WHISPER_COMPUTE_TYPE,
        settings.WHISPER_BEAM_SIZE,
        settings.WHISPER_VAD_FILTER,
        settings.WHISPER_CPU_THREADS or "auto",
    )
except ImportError:
    _log.warning("faster-whisper not installed — transcription unavailable")
    _whisper_model = None
except Exception as _exc:
    _log.error(
        "faster-whisper failed to load | %s: %s — transcription unavailable",
        type(_exc).__name__,
        _exc,
    )
    _whisper_model = None


# ---------------------------------------------------------------------------
# Stage helpers — each represents one discrete pipeline stage
# ---------------------------------------------------------------------------

def _extract_audio(video_path: str) -> str:
    """
    Extract audio from a video file using ffmpeg.

    Converts to mono 16 kHz WAV — the format faster-whisper expects.
    Returns the path to a temporary WAV file.

    The CALLER must delete this file after use (use try/finally).

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero return code.
    """
    _log.info("audio_extract | start | source=%s", video_path)
    t0 = time.monotonic()

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    audio_path = tmp.name

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        stderr_text = result.stderr.decode(errors="replace")
        raise RuntimeError(
            f"ffmpeg failed (exit={result.returncode}) for '{video_path}'.\n"
            f"stderr: {stderr_text[:1000]}"
        )

    size_kb = os.path.getsize(audio_path) // 1024
    _log.info(
        "audio_extract | done | output=%s size_kb=%d elapsed=%.1fs",
        audio_path,
        size_kb,
        time.monotonic() - t0,
    )
    return audio_path


def _transcribe_audio(audio_path: str) -> str:
    """
    Transcribe a WAV file using faster-whisper.

    Language is read from settings.WHISPER_LANGUAGE (default: "en").
    Beam size and VAD filter are read from settings.

    Returns:
        Full transcript as a single stripped string.

    Raises:
        RuntimeError: If faster-whisper is not installed or failed to load.
    """
    if _whisper_model is None:
        raise RuntimeError(
            "faster-whisper is not available. "
            "Ensure it is installed in the worker image."
        )

    _log.info(
        "transcribe | start | lang=%s model=%s beam_size=%d vad=%s",
        settings.WHISPER_LANGUAGE,
        settings.WHISPER_MODEL,
        settings.WHISPER_BEAM_SIZE,
        settings.WHISPER_VAD_FILTER,
    )
    t0 = time.monotonic()

    segments, info = _whisper_model.transcribe(
        audio_path,
        language=settings.WHISPER_LANGUAGE,
        task="transcribe",
        beam_size=settings.WHISPER_BEAM_SIZE,
        vad_filter=settings.WHISPER_VAD_FILTER,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()

    _log.info(
        "transcribe | done | detected_lang=%s chars=%d elapsed=%.1fs",
        getattr(info, "language", "unknown"),
        len(text),
        time.monotonic() - t0,
    )
    return text


def _chunk_transcript(session: Session, video_id: int, text: str) -> List:
    """
    Split the transcript into sentence-aware overlapping chunks and persist them.

    Deletes any pre-existing chunks first so the stage is idempotent on retry.

    Returns:
        The list of persisted TranscriptChunk DB objects.
    """
    _log.info("chunking | start | video_id=%s", video_id)
    t0 = time.monotonic()

    deleted = delete_chunks_for_video(session, video_id)
    if deleted:
        _log.info(
            "chunking | cleared %d previous chunks | video_id=%s",
            deleted,
            video_id,
        )

    raw_chunks = split_into_chunks(text)
    db_chunks = create_chunks(session, video_id=video_id, chunks=raw_chunks)

    _log.info(
        "chunking | done | video_id=%s count=%d elapsed=%.1fs",
        video_id,
        len(db_chunks),
        time.monotonic() - t0,
    )
    return db_chunks


def _generate_embeddings(session: Session, video_id: int, chunk_count: int) -> None:
    """
    Generate and persist embedding vectors for all un-embedded chunks.

    Uses the model configured in settings.EMBEDDING_MODEL_NAME.
    Safe to call more than once — already-embedded chunks are skipped.
    """
    _log.info(
        "embedding | start | video_id=%s chunks=%d model=%s",
        video_id,
        chunk_count,
        settings.EMBEDDING_MODEL_NAME,
    )
    t0 = time.monotonic()

    embed_all_chunks(session, video_id)

    _log.info(
        "embedding | done | video_id=%s elapsed=%.1fs",
        video_id,
        time.monotonic() - t0,
    )


# ---------------------------------------------------------------------------
# Status helper
# ---------------------------------------------------------------------------

def _set_status(session: Session, video: Video, status: str) -> None:
    """Persist a status change immediately (commit-per-stage for observability)."""
    video.status = status
    session.add(video)
    session.commit()


# ---------------------------------------------------------------------------
# Main pipeline job
# ---------------------------------------------------------------------------

def process_video_pipeline(video_id: int) -> None:
    """
    Full video processing pipeline executed by the RQ worker.

    Status transitions:
        queued → processing  (audio extraction)
               → transcribing (Whisper)
               → embedding   (chunking + embeddings)
               → completed
        Any exception → failed (error_message saved to DB)

    The job opens its own DB session — it does NOT rely on FastAPI
    request-scoped dependencies.

    Args:
        video_id: Primary key of the Video row to process.
    """
    _log.info("pipeline | start | video_id=%s", video_id)
    t_total = time.monotonic()
    audio_path: str | None = None
    s3_tmp_path: str | None = None

    with Session(engine) as session:
        video = session.get(Video, video_id)
        if video is None:
            _log.error(
                "pipeline | aborted | video_id=%s | not found in DB",
                video_id,
            )
            return

        try:
            # ── Stage 1 — audio extraction ─────────────────────────────────
            _set_status(session, video, "processing")
            _log.info("pipeline | stage=processing | video_id=%s", video_id)

            # Resolve the video file: S3 (production) or local path (dev/legacy)
            if video.s3_key:
                _log.info(
                    "pipeline | downloading from S3 | video_id=%s key=%s",
                    video_id,
                    video.s3_key,
                )
                s3_tmp_path = download_file_from_s3(video.s3_key)
                video_source = s3_tmp_path
            elif video.file_path and os.path.exists(video.file_path):
                video_source = video.file_path
            else:
                raise RuntimeError(
                    f"Video {video_id} has no accessible file "
                    f"(s3_key={video.s3_key!r}, file_path={video.file_path!r})."
                )

            _log.info(
                "pipeline | file resolved | video_id=%s path=%s",
                video_id,
                video_source,
            )

            t_ffmpeg = time.monotonic()
            audio_path = _extract_audio(video_source)
            _log.info(
                "pipeline | audio extracted | video_id=%s elapsed=%.1fs",
                video_id,
                time.monotonic() - t_ffmpeg,
            )

            # ── Stage 2 — Whisper transcription ───────────────────────────
            _set_status(session, video, "transcribing")
            _log.info("pipeline | stage=transcribing | video_id=%s", video_id)

            t_whisper = time.monotonic()
            text = _transcribe_audio(audio_path)
            _log.info(
                "pipeline | transcription done | video_id=%s chars=%d elapsed=%.1fs",
                video_id,
                len(text),
                time.monotonic() - t_whisper,
            )

            # Persist transcript (upsert: overwrite if a previous run saved one)
            existing = get_transcript_by_video_id(session, video_id)
            if existing is not None:
                existing.content = text
                existing.source = "whisper"
                session.add(existing)
            else:
                create_transcript(
                    session,
                    video_id=video_id,
                    content=text,
                    source="whisper",
                )
            session.commit()
            _log.info("pipeline | transcript saved | video_id=%s", video_id)

            # ── Stage 3 — chunking + embedding ────────────────────────────
            _set_status(session, video, "embedding")
            _log.info("pipeline | stage=embedding | video_id=%s", video_id)

            db_chunks = _chunk_transcript(session, video_id, text)
            _generate_embeddings(session, video_id, len(db_chunks))

            # ── Done ──────────────────────────────────────────────────────
            _set_status(session, video, "completed")
            _log.info(
                "pipeline | completed | video_id=%s total_elapsed=%.1fs "
                "(model=%s beam=%d lang=%s)",
                video_id,
                time.monotonic() - t_total,
                settings.WHISPER_MODEL,
                settings.WHISPER_BEAM_SIZE,
                settings.WHISPER_LANGUAGE,
            )

        except Exception as exc:
            elapsed = time.monotonic() - t_total
            _log.exception(
                "pipeline | FAILED | video_id=%s elapsed=%.1fs | %s: %s",
                video_id,
                elapsed,
                type(exc).__name__,
                exc,
            )
            # Roll back any partial changes, then mark the video as failed.
            try:
                session.rollback()
                v = session.get(Video, video_id)
                if v:
                    v.status = "failed"
                    # Truncate to 500 chars — error_message is for diagnostics,
                    # not a full stack trace (that lives in the worker logs).
                    v.error_message = f"{type(exc).__name__}: {str(exc)[:500]}"
                    session.add(v)
                    session.commit()
            except Exception as update_exc:
                _log.error(
                    "pipeline | could not persist status=failed | video_id=%s | %s",
                    video_id,
                    update_exc,
                )

        finally:
            # Always clean up temp files regardless of success or failure.
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            if s3_tmp_path and os.path.exists(s3_tmp_path):
                os.remove(s3_tmp_path)
