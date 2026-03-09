import os
import subprocess
import tempfile

from sqlmodel import Session, select

from app.models.transcript import Transcript
from app.models.video import Video


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
# Whisper transcription
# ---------------------------------------------------------------------------

# Model is loaded once at module import time to avoid the multi-second
# reload penalty on every transcription request. "medium" is chosen because
# Hebrew is under-represented in Whisper's training data and smaller models
# produce noticeably degraded output. fp16=False is required on CPU (Windows).
_WHISPER_MODEL_NAME = "medium"

try:
    import whisper as _whisper
    _whisper_model = _whisper.load_model(_WHISPER_MODEL_NAME)
except Exception:  # whisper not installed — server still starts
    _whisper_model = None


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file using OpenAI Whisper (Hebrew only).

    The model is loaded once at module level (_whisper_model) and reused
    across calls. language="he" and task="transcribe" are fixed to prevent
    auto-detection from switching to the wrong language.

    Args:
        audio_path:  Path to a WAV file produced by extract_audio().

    Returns:
        The full transcript as a single string with leading/trailing
        whitespace stripped.

    Raises:
        RuntimeError: If the whisper package is not installed.
    """
    if _whisper_model is None:
        raise RuntimeError(
            "openai-whisper is not installed. Run: pip install openai-whisper"
        )

    result = _whisper_model.transcribe(audio_path, language="he", task="transcribe", fp16=False)
    return result["text"].strip()


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
