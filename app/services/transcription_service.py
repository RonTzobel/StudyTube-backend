"""
transcription_service.py — Audio transcription via the OpenAI Whisper API.

Why external API instead of local faster-whisper?
    Local faster-whisper / CTranslate2 stalls indefinitely on EC2 CPU-only
    Docker instances (near-zero CPU, no exception, no progress) due to OpenMP
    thread scheduling issues inside the container. The OpenAI Whisper API
    eliminates all local inference and runtime stall risk.

Why still split into chunks?
    The OpenAI Whisper API accepts files up to 25 MB per request.
    A 5-minute mono 16 kHz 16-bit WAV is ~9.2 MB — safely under the limit.
    A 53-minute lecture at the same spec is ~97 MB, so splitting is required
    for any real lecture video. The 5-minute chunk size is kept from the
    previous local approach and works identically here.

Public interface:
    transcribe_audio_file(audio_path: str) -> str
"""

import logging
import os
import subprocess
import tempfile
import time
from typing import List

from app.config.settings import settings

_log = logging.getLogger(__name__)

# OpenAI Whisper API hard limit is 25 MB per request.
# 5-minute mono 16 kHz 16-bit WAV ≈ 9.2 MB — one chunk fits with margin.
_CHUNK_SECONDS = 300


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _split_audio_into_chunks(audio_path: str) -> List[str]:
    """
    Split a WAV file into fixed-length segments using ffmpeg.

    Segments are written to a temp directory as chunk_000.wav, chunk_001.wav …
    The CALLER must delete all returned paths when done (use try/finally).

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero return code.
    """
    chunk_dir = tempfile.mkdtemp(prefix="transcription_chunks_")
    pattern = os.path.join(chunk_dir, "chunk_%03d.wav")

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-f", "segment",
            "-segment_time", str(_CHUNK_SECONDS),
            "-c", "copy",
            pattern,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        stderr_text = result.stderr.decode(errors="replace")
        raise RuntimeError(
            f"ffmpeg segmentation failed (exit={result.returncode}): "
            f"{stderr_text[:500]}"
        )

    chunks = sorted(
        os.path.join(chunk_dir, f)
        for f in os.listdir(chunk_dir)
        if f.endswith(".wav")
    )
    _log.info(
        "transcription | audio split | chunks=%d chunk_s=%d",
        len(chunks),
        _CHUNK_SECONDS,
    )
    return chunks


def _call_whisper_api(client, chunk_path: str) -> str:
    """
    Upload one audio chunk to the OpenAI Whisper API and return the transcript.

    response_format="text" makes the SDK return a plain string directly.
    The isinstance guard handles both the string response and any SDK version
    that wraps it in a Transcription object.
    """
    with open(chunk_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=settings.OPENAI_TRANSCRIPTION_MODEL,
            file=audio_file,
            language="en",
            response_format="text",
        )
    # openai>=1.0.0 with response_format="text" returns a bare str.
    # Guard against SDK versions that return an object with .text.
    text = response if isinstance(response, str) else response.text
    return text.strip()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def transcribe_audio_file(audio_path: str) -> str:
    """
    Transcribe a WAV audio file using the OpenAI Whisper API.

    The file is split into _CHUNK_SECONDS-length segments before upload so
    that no single request exceeds the 25 MB API size limit. Each chunk is
    transcribed independently and the results are joined in order.

    Args:
        audio_path: Absolute path to a local WAV file (mono 16 kHz, from ffmpeg).

    Returns:
        Full transcript as a single stripped string.

    Raises:
        RuntimeError: If OPENAI_API_KEY is missing or any API call fails.
                      The caller (jobs.py) catches this and marks the video
                      as failed with error_message set.
    """
    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Add it to .env to enable transcription."
        )

    # Deferred import — openai is available in both backend and worker images,
    # but importing at module level would load it even when not used.
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.OPENAI_API_KEY,
        timeout=settings.TRANSCRIPTION_TIMEOUT,
    )

    _log.info(
        "transcription | start | model=%s file=%s timeout=%ds",
        settings.OPENAI_TRANSCRIPTION_MODEL,
        audio_path,
        settings.TRANSCRIPTION_TIMEOUT,
    )
    t_total = time.monotonic()

    chunk_paths = _split_audio_into_chunks(audio_path)
    chunk_texts: List[str] = []

    try:
        for i, chunk_path in enumerate(chunk_paths):
            t_chunk = time.monotonic()
            size_kb = os.path.getsize(chunk_path) // 1024

            _log.info(
                "transcription | chunk start | %d/%d | size_kb=%d",
                i + 1,
                len(chunk_paths),
                size_kb,
            )

            try:
                text = _call_whisper_api(client, chunk_path)
            except Exception as exc:
                raise RuntimeError(
                    f"OpenAI Whisper API failed on chunk {i + 1}/{len(chunk_paths)}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            chunk_texts.append(text)
            _log.info(
                "transcription | chunk done | %d/%d | chars=%d elapsed=%.1fs",
                i + 1,
                len(chunk_paths),
                len(text),
                time.monotonic() - t_chunk,
            )

    finally:
        # Always clean up temp chunk files, even if a chunk fails mid-way.
        for p in chunk_paths:
            if os.path.exists(p):
                os.remove(p)
        chunk_dir = os.path.dirname(chunk_paths[0]) if chunk_paths else None
        if chunk_dir and os.path.isdir(chunk_dir):
            try:
                os.rmdir(chunk_dir)
            except OSError:
                pass  # non-empty on partial failure — leave it, don't mask the real error

    full_text = " ".join(t for t in chunk_texts if t).strip()
    _log.info(
        "transcription | done | chunks=%d total_chars=%d elapsed=%.1fs",
        len(chunk_paths),
        len(full_text),
        time.monotonic() - t_total,
    )
    return full_text
