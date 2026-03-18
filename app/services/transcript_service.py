import logging
import os
import subprocess
import tempfile
import time

_log = logging.getLogger(__name__)

from sqlmodel import Session, select

from app.config.settings import settings
from app.database.session import engine
from app.models.transcript import Transcript
from app.models.video import Video

# Imported here (not at top of module) to avoid a potential circular import
# between transcript_service ↔ chunk_service / embedding_service.
# Both chunk and embedding services only import from models — no circular risk.
from app.services.chunk_service import create_chunks, split_into_chunks
from app.services.embedding_service import embed_all_chunks


# ---------------------------------------------------------------------------
# Video / transcript lookups
# ---------------------------------------------------------------------------

def get_video_by_id(session: Session, video_id: int) -> Video | None:
    """
    Look up a video by its primary key.

    Used by the transcript endpoint to confirm the video exists before
    querying for its transcript. Returning None (instead of raising)
    keeps the decision about HTTP status codes in the router.
    """
    return session.get(Video, video_id)


def get_transcript_by_video_id(session: Session, video_id: int) -> Transcript | None:
    """
    Return the transcript for a given video, or None if it doesn't exist yet.

    The transcripts table has a unique constraint on video_id (one video →
    one transcript), so .first() is the correct call here — there can never
    be more than one result.

    Returning None (instead of raising) keeps the decision about what HTTP
    status code to send in the router, where it belongs.
    """
    return session.exec(
        select(Transcript).where(Transcript.video_id == video_id)
    ).first()


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: str) -> str:
    """
    Extract the audio track from a video file using ffmpeg.

    The output is a temporary mono WAV file sampled at 16 kHz —
    the exact format Whisper expects. The temp file is written to the
    OS temp directory (e.g. C:/Users/.../AppData/Local/Temp on Windows).

    The CALLER is responsible for deleting the returned file after use.
    Wrap the call in a try/finally block to guarantee cleanup even on error.

    Args:
        video_path: Absolute path to the video file saved on disk.

    Returns:
        Path to the extracted WAV file.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero return code.
        FileNotFoundError: If ffmpeg is not installed / not on PATH.
    """
    # Create a temp file so we have a guaranteed unique path.
    # delete=False because ffmpeg writes to it after we close the handle.
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    audio_path = tmp.name

    result = subprocess.run(
        [
            "ffmpeg", "-y",          # -y: overwrite output without asking
            "-i", video_path,        # input: the uploaded video file
            "-vn",                   # drop the video stream entirely
            "-acodec", "pcm_s16le",  # output codec: uncompressed 16-bit PCM
            "-ar", "16000",          # sample rate: 16 kHz (Whisper's sweet spot)
            "-ac", "1",              # channels: mono
            audio_path,              # output: the temp WAV file
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        # Clean up the empty temp file before raising
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise RuntimeError(
            f"ffmpeg failed to extract audio from '{video_path}'.\n"
            f"stderr: {result.stderr.decode(errors='replace')}"
        )

    return audio_path


# ---------------------------------------------------------------------------
# Whisper transcription  (faster-whisper)
# ---------------------------------------------------------------------------

# Model is loaded once at module import time to avoid the multi-second
# reload penalty on every transcription request.
# device="cpu" + compute_type="int8" are safe defaults for Windows / no-GPU setups.
# cpu_threads=0 lets CTranslate2 pick the thread count automatically (all available cores).
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
    _log.error("faster-whisper model failed to load (%s: %s) — transcription unavailable",
               type(_exc).__name__, _exc)
    _whisper_model = None


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file using faster-whisper (Hebrew only).

    The model is loaded once at module level (_whisper_model) and reused
    across calls. language="he" and task="transcribe" are fixed to prevent
    auto-detection from switching to the wrong language.

    Performance knobs (all controlled via settings / .env):
      beam_size   — 1 (greedy) is 2–3× faster than the default of 5 with
                    minimal quality loss for structured lecture content.
      vad_filter  — skips silence and non-speech frames, saving 10–30 % of
                    processing time for typical lecture videos.

    faster-whisper returns a generator of segments; we join them into a
    single string to match the previous openai-whisper behavior.

    Args:
        audio_path:  Path to a WAV file produced by extract_audio().

    Returns:
        The full transcript as a single string with leading/trailing
        whitespace stripped.

    Raises:
        RuntimeError: If faster-whisper is not installed.
    """
    if _whisper_model is None:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        )

    # segments is a generator — consume it once to build the full text.
    segments, _info = _whisper_model.transcribe(
        audio_path,
        language="he",
        task="transcribe",
        beam_size=settings.WHISPER_BEAM_SIZE,
        vad_filter=settings.WHISPER_VAD_FILTER,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


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

    This function does NOT check for an existing transcript — the caller
    (the router) is responsible for the idempotency check so the error
    message and HTTP status code stay in the right layer.

    Args:
        session:   The database session injected by FastAPI.
        video_id:  The primary key of the video this transcript belongs to.
        content:   The full transcript text returned by Whisper.
        source:    How the transcript was produced — "whisper" by default.

    Returns:
        The newly created, database-refreshed Transcript object.
    """
    transcript = Transcript(video_id=video_id, content=content, source=source)
    session.add(transcript)
    session.commit()
    session.refresh(transcript)
    return transcript


# ---------------------------------------------------------------------------
# Background transcription job
# ---------------------------------------------------------------------------

def run_transcription_background(video_id: int) -> None:
    """
    Full transcription pipeline executed as a FastAPI BackgroundTask.

    This function is intentionally decoupled from the HTTP request lifecycle:
    it opens its own database session via the shared engine so it can safely
    run after the 202 response has already been sent to the client.

    Status transitions:
        processing → done    (success)
        processing → failed  (any exception)

    The caller (router) is responsible for setting status="processing" and
    persisting that change *before* enqueueing this task, so the client
    always sees a consistent "processing" state the moment the 202 arrives.

    Args:
        video_id: Primary key of the Video row to transcribe.
    """
    _log.info("transcription started | video_id=%s", video_id)
    t0 = time.monotonic()
    audio_path: str | None = None

    with Session(engine) as session:
        video = session.get(Video, video_id)
        if video is None:
            # Should never happen — router verified existence before enqueueing.
            _log.error("transcription aborted | video_id=%s | video not found in DB", video_id)
            return

        try:
            # Stage 1 — ffmpeg audio extraction
            t_ffmpeg = time.monotonic()
            audio_path = extract_audio(video.file_path)
            _log.info(
                "audio extracted | video_id=%s | elapsed=%.1fs",
                video_id, time.monotonic() - t_ffmpeg,
            )

            # Stage 2 — Whisper transcription (the slow stage for long videos)
            t_whisper = time.monotonic()
            text = transcribe_audio(audio_path)
            _log.info(
                "audio transcribed | video_id=%s | chars=%d | elapsed=%.1fs",
                video_id, len(text), time.monotonic() - t_whisper,
            )

            # Stage 3 — persist transcript
            t_db = time.monotonic()
            # Guard against a race where two requests somehow both passed the
            # "processing" check. The transcripts table has a unique constraint
            # on video_id, so the second insert would fail; updating in place
            # is safer.
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
            _log.info(
                "transcript saved | video_id=%s | elapsed=%.1fs",
                video_id, time.monotonic() - t_db,
            )

            # Stage 4 — chunking
            video.status = "indexing"
            session.add(video)
            session.commit()

            t_chunk = time.monotonic()
            raw_chunks = split_into_chunks(text)
            db_chunks = create_chunks(session, video_id=video_id, chunks=raw_chunks)
            _log.info(
                "chunks created | video_id=%s | count=%d | elapsed=%.1fs",
                video_id, len(db_chunks), time.monotonic() - t_chunk,
            )

            # Stage 5 — embedding
            t_embed = time.monotonic()
            embed_all_chunks(session, video_id)
            _log.info(
                "embeddings created | video_id=%s | elapsed=%.1fs",
                video_id, time.monotonic() - t_embed,
            )

            video.status = "ready"
            session.add(video)
            session.commit()

            elapsed = time.monotonic() - t0
            _log.info(
                "video ready | video_id=%s | total=%.1fs "
                "(model=%s beam=%s vad=%s)",
                video_id, elapsed,
                settings.WHISPER_MODEL,
                settings.WHISPER_BEAM_SIZE,
                settings.WHISPER_VAD_FILTER,
            )

        except Exception as exc:
            elapsed = time.monotonic() - t0
            _log.exception(
                "transcription failed | video_id=%s | elapsed=%.1fs | %s: %s",
                video_id, elapsed, type(exc).__name__, exc,
            )
            # Roll back any partial work from the failed transaction, then
            # mark the video as failed so the client does not poll forever.
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
                    video_id, update_exc,
                )

        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
