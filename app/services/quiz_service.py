"""
quiz_service.py — Generate multiple-choice quiz questions from transcript chunks.

Design decisions:
  - Chunks are sampled EVENLY by chunk_index rather than retrieved by similarity.
    Quiz generation needs broad transcript coverage (whole video), not a narrow
    slice relevant to a single query.
  - OpenAI is asked to return strict JSON via response_format=json_object.
    This eliminates fragile string parsing — we just json.loads() the reply.
  - The system prompt explicitly forbids the LLM from inventing facts outside
    the provided excerpts, keeping all questions grounded in the transcript.
"""

import json
from typing import List

from sqlmodel import Session

from app.config.settings import settings
from app.core.language import detect_language, language_instruction
from app.models.chunk import TranscriptChunk
from app.schemas.quiz import QuizQuestion, QuizResponse
from app.services.retrieval_service import get_embedded_chunks


# ---------------------------------------------------------------------------
# OpenAI client — same lazy-import pattern used in qa_service.py
# ---------------------------------------------------------------------------

def _get_openai_client():
    """
    Return an initialised OpenAI client.

    Raises:
        RuntimeError: If the openai package is not installed.
        RuntimeError: If OPENAI_API_KEY is not set.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "The 'openai' package is not installed. "
            "Run: pip install openai"
        )

    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file: OPENAI_API_KEY=sk-..."
        )

    return OpenAI(api_key=settings.OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Chunk sampling
#
# For quiz generation we want broad coverage of the transcript, not the most
# similar chunks to a single query. We achieve this by picking chunks that
# are evenly spaced by their position (chunk_index) in the transcript.
#
# Example — 20 chunks, top_k=5:
#   selected indices ≈ [0, 4, 9, 14, 19]  (one per ~quarter of the transcript)
# ---------------------------------------------------------------------------

def _sample_chunks_evenly(
    chunks: List[TranscriptChunk],
    top_k: int,
) -> List[TranscriptChunk]:
    """
    Pick up to top_k chunks that are evenly spaced across the transcript.

    If the video has fewer chunks than top_k, all chunks are returned.
    Chunks must already be sorted by chunk_index ascending.
    """
    if len(chunks) <= top_k:
        return chunks

    # Build evenly spaced indices from 0 to len-1, then pick those chunks.
    step = (len(chunks) - 1) / (top_k - 1)
    indices = {round(i * step) for i in range(top_k)}
    return [chunks[i] for i in sorted(indices)]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_MESSAGE_BASE = (
    "You are a quiz generator for a video learning platform. "
    "You will be given excerpts from a video transcript. "
    "Generate multiple-choice quiz questions based ONLY on the information "
    "in the provided transcript excerpts. "
    "Do NOT invent facts, names, numbers, or concepts that are not mentioned "
    "in the excerpts. "
    "Each question must have exactly 4 answer options labeled A, B, C, D. "
    "Only one option is correct. "
    "Return valid JSON only — no markdown, no extra text."
)


def _build_quiz_prompt(chunks: List[TranscriptChunk], num_questions: int, lang: str) -> str:
    """
    Build the user-turn message for quiz generation.

    lang is detected from the transcript chunks so the directive is based on
    the lecture language, not a user question (there is none for quiz generation).
    """
    context_blocks = "\n\n".join(
        f"[Excerpt {i + 1}]\n{chunk.content}"
        for i, chunk in enumerate(chunks)
    )

    return (
        f"Transcript excerpts:\n"
        f"{context_blocks}\n\n"
        f"Generate exactly {num_questions} multiple-choice questions based on "
        f"the excerpts above.\n\n"
        f"{language_instruction(lang)}\n"
        f"This applies to ALL parts of the output: question text, all answer "
        f"options (A, B, C, D), and the correct_answer field.\n\n"
        f"Return your response as a JSON object in this exact format:\n"
        f'{{\n'
        f'  "questions": [\n'
        f'    {{\n'
        f'      "question": "Question text here?",\n'
        f'      "options": ["A. first option", "B. second option", "C. third option", "D. fourth option"],\n'
        f'      "correct_answer": "A. first option"\n'
        f'    }}\n'
        f'  ]\n'
        f'}}'
    )


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_quiz_response(raw_json: str) -> List[QuizQuestion]:
    """
    Parse OpenAI's JSON response into a list of QuizQuestion objects.

    Args:
        raw_json: The raw string returned by OpenAI (should be valid JSON).

    Returns:
        A list of QuizQuestion objects.

    Raises:
        ValueError: If the JSON is malformed or missing required fields.
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenAI returned invalid JSON: {exc}") from exc

    raw_questions = data.get("questions", [])
    if not raw_questions:
        raise ValueError(
            "OpenAI returned no questions. "
            "The transcript may be too short to generate meaningful questions."
        )

    questions: List[QuizQuestion] = []
    for i, q in enumerate(raw_questions):
        # Validate each field — give the caller a clear error if the LLM misbehaved
        if not isinstance(q.get("question"), str):
            raise ValueError(f"Question {i + 1} is missing a 'question' string.")
        options = q.get("options", [])
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError(f"Question {i + 1} must have exactly 4 options.")
        if not isinstance(q.get("correct_answer"), str):
            raise ValueError(f"Question {i + 1} is missing a 'correct_answer' string.")

        questions.append(
            QuizQuestion(
                question=q["question"],
                options=options,
                correct_answer=q["correct_answer"],
            )
        )

    return questions


# ---------------------------------------------------------------------------
# Main quiz generation function
# ---------------------------------------------------------------------------

def generate_quiz(
    session: Session,
    video_id: int,
    num_questions: int = 5,
    top_k: int = 10,
    model: str = "gpt-4o-mini",
) -> QuizResponse:
    """
    Generate a multiple-choice quiz grounded in a video's transcript.

    Steps:
      1. Load all embedded chunks for the video.
      2. Sample top_k chunks evenly across the transcript for broad coverage.
      3. Build a prompt asking OpenAI for num_questions MC questions.
      4. Call OpenAI with response_format=json_object for safe parsing.
      5. Parse the JSON and return a QuizResponse.

    Args:
        session:       DB session injected by FastAPI.
        video_id:      The video to generate a quiz for.
        num_questions: How many questions to generate.
        top_k:         How many transcript chunks to pass as context.
        model:         OpenAI model. gpt-4o-mini is fast and cheap.

    Returns:
        QuizResponse containing the generated questions.

    Raises:
        ValueError:   If no embedded chunks exist, or if OpenAI returns bad JSON.
        RuntimeError: If openai is not installed or OPENAI_API_KEY is missing.
    """
    # Step 1 — load all embedded chunks (sorted by chunk_index)
    all_chunks = get_embedded_chunks(session, video_id)
    if not all_chunks:
        raise ValueError(
            f"No embedded chunks found for video {video_id}. "
            "Transcribe the video first (POST /transcribe) and wait for status 'ready'."
        )

    # Step 2 — sample evenly for broad transcript coverage
    sampled = _sample_chunks_evenly(all_chunks, top_k)

    # Detect language from the transcript content.
    # There is no user question for quiz generation, so the transcript itself
    # is the only language signal.
    transcript_sample = " ".join(c.content for c in sampled)
    lang = detect_language(transcript_sample)

    # Step 3 — build the prompt with an explicit language directive
    user_message = _build_quiz_prompt(sampled, num_questions, lang)

    # Step 4 — call OpenAI; language instruction is also injected into the
    # system message so it is reinforced at both the system and user levels.
    system_message = f"{_SYSTEM_MESSAGE_BASE}\n{language_instruction(lang)}"
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.4,          # slightly higher than /ask for question variety
        max_tokens=2048,
        response_format={"type": "json_object"},   # guarantees valid JSON output
    )

    raw_json = response.choices[0].message.content.strip()

    # Step 5 — parse and return
    questions = _parse_quiz_response(raw_json)

    return QuizResponse(
        video_id=video_id,
        num_questions=len(questions),
        questions=questions,
    )
