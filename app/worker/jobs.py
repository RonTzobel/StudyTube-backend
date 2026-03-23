"""
jobs.py — RQ job functions executed by the worker process.

ALL heavy computation lives here:
  - ffmpeg audio extraction
  - faster-whisper transcription
  - chunking
  - sentence-transformers embedding

FastAPI NEVER imports this module, so the Whisper model and the embedding
model are loaded only in the worker process, not in the API server.

Each job function opens its own database session via the shared SQLAlchemy
engine — it does NOT use FastAPI request-scoped dependencies.
"""

import logging
import os
import subprocess
import tempfile
import time

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
        "faster-whisper loaded: model=%s device=%s compute=%s "
        "beam_size=%s vad=%s cpu_threads=%s",
        settings.WHISPER_MODEL,
        settings.WHISPER_DEVICE,
        settings.WHISPER_COMPUTE_TYPE,
        settings.WHISPER_BEAM_SIZE,
        settings.WHISPER_VAD_FILTER,
        settings.WHISPER_CPU_THREADS or "auto",
    )
except ImportError:
    _log.warning("faster-whisper is not installed — transcription unavailable")
    _whisper_model = None
except Exception as _exc:
    _log.error(
        "faster-whisper model failed to load (%s: %s) — transcription unavailable",
        type(_exc).__name__,
        _exc,
    )
    _whisper_model = None


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def _extract_audio(video_path: str) -> str:
    """
    Extract audio track from a video file using ffmpeg.

    Returns the path to a temporary mono 16 kHz WAV file.
    The CALLER must delete this file after use (wrap in try/finally).

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero return code.
    """
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
        raise RuntimeError(
            f"ffmpeg failed to extract audio from '{video_path}'.\n"
            f"stderr: {result.stderr.decode(errors='replace')}"
        )

    return audio_path


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

def _transcribe_audio(audio_path: str) -> str:
    """
    Transcribe a WAV file using faster-whisper.

    Language is fixed to Hebrew ("he") to prevent auto-detection errors.
    Performance knobs are controlled by settings (beam_size, vad_filter).

    Returns:
        Full transcript as a single stripped string.

    Raises:
        RuntimeError: If faster-whisper is not installed.
    """
    if _whisper_model is None:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        )

    segments, _info = _whisper_model.transcribe(
        audio_path,
        language="he",
        task="transcribe",
        beam_size=settings.WHISPER_BEAM_SIZE,
        vad_filter=settings.WHISPER_VAD_FILTER,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


# ---------------------------------------------------------------------------
# Main pipeline job
# ---------------------------------------------------------------------------

def process_video_pipeline(video_id: int) -> None:
    """
    Full video processing pipeline executed by the RQ worker.

    Status transitions:
        queued → processing → transcribed → indexing → ready
        queued → processing → failed  (on any exception)

    The job opens its own DB session — it does NOT rely on FastAPI
    request-scoped dependencies.

    Args:
        video_id: Primary key of the Video row to process.
    """
    _log.info("pipeline start | video_id=%s", video_id)
    t0 = time.monotonic()
    audio_path: str | None = None
    s3_tmp_path: str | None = None   # temp local copy of an S3 video

    with Session(engine) as session:
        video = session.get(Video, video_id)
        if video is None:
            _log.error(
                "pipeline aborted | video_id=%s | video not found in DB", video_id
            )
            return

        try:
            # ── Stage 1 — mark as processing ──────────────────────────────
            video.status = "processing"
            session.add(video)
            session.commit()

            # ── Stage 2 — resolve video source ────────────────────────────
            # New uploads live in S3; legacy uploads may have a local path.
            if video.s3_key:
                s3_tmp_path = download_file_from_s3(video.s3_key)
                video_source = s3_tmp_path
            elif video.file_path and os.path.exists(video.file_path):
                video_source = video.file_path
            else:
                raise RuntimeError(
                    f"Video {video_id} has no accessible file "
                    f"(s3_key={video.s3_key!r}, file_path={video.file_path!r})."
                )

            # ── Stage 3 — ffmpeg audio extraction ─────────────────────────
            t_ffmpeg = time.monotonic()
            audio_path = _extract_audio(video_source)
            _log.info(
                "audio extracted | video_id=%s | elapsed=%.1fs",
                video_id,
                time.monotonic() - t_ffmpeg,
            )

            # ── Stage 3 — Whisper transcription ───────────────────────────
            t_whisper = time.monotonic()
            text = _transcribe_audio(audio_path)
            _log.info(
                "transcribed | video_id=%s | chars=%d | elapsed=%.1fs",
                video_id,
                len(text),
                time.monotonic() - t_whisper,
            )

            # ── Stage 4 — persist transcript ──────────────────────────────
            existing = get_transcript_by_video_id(session, video_id)
            if existing is not None:
                existing.content = text
                existing.source = "whisper"
                session.add(existing)
            else:
                create_transcript(session, video_id=video_id, content=text, source="whisper")

            video.status = "transcribed"
            session.add(video)
            session.commit()
            _log.info("transcript saved | video_id=%s", video_id)

            # ── Stage 5 — chunking ────────────────────────────────────────
            video.status = "indexing"
            session.add(video)
            session.commit()

            t_chunk = time.monotonic()
            # Delete any chunks from a previous run before inserting new ones.
            # This makes the stage idempotent: if the pipeline failed partway
            # through indexing and is retried, we won't accumulate duplicates.
            deleted = delete_chunks_for_video(session, video_id)
            if deleted:
                _log.info(
                    "chunks cleared (re-run) | video_id=%s | deleted=%d",
                    video_id, deleted,
                )
            raw_chunks = split_into_chunks(text)
            db_chunks = create_chunks(session, video_id=video_id, chunks=raw_chunks)
            _log.info(
                "chunks created | video_id=%s | count=%d | elapsed=%.1fs",
                video_id,
                len(db_chunks),
                time.monotonic() - t_chunk,
            )

            # ── Stage 6 — embedding ───────────────────────────────────────
            t_embed = time.monotonic()
            embed_all_chunks(session, video_id)
            _log.info(
                "embeddings done | video_id=%s | elapsed=%.1fs",
                video_id,
                time.monotonic() - t_embed,
            )

            # ── Done ──────────────────────────────────────────────────────
            video.status = "ready"
            session.add(video)
            session.commit()

            _log.info(
                "pipeline done | video_id=%s | total=%.1fs "
                "(model=%s beam=%s vad=%s)",
                video_id,
                time.monotonic() - t0,
                settings.WHISPER_MODEL,
                settings.WHISPER_BEAM_SIZE,
                settings.WHISPER_VAD_FILTER,
            )

        except Exception as exc:
            _log.exception(
                "pipeline failed | video_id=%s | elapsed=%.1fs | %s: %s",
                video_id,
                time.monotonic() - t0,
                type(exc).__name__,
                exc,
            )
            # Roll back any partial changes, then mark the video as failed
            # so the client does not poll forever.
            try:
                session.rollback()
                v = session.get(Video, video_id)
                if v:
                    v.status = "failed"
                    session.add(v)
                    session.commit()
            except Exception as update_exc:
                _log.error(
                    "could not persist status=failed | video_id=%s | %s",
                    video_id,
                    update_exc,
                )

        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            # Clean up the temp S3 download (only created for S3-backed videos)
            if s3_tmp_path and os.path.exists(s3_tmp_path):
                os.remove(s3_tmp_path)
