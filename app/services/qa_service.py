import logging
import time
from typing import List

from sqlmodel import Session

from app.config.settings import settings
from app.schemas.qa import AskResponse
from app.schemas.retrieval import RetrievedChunkRead
from app.services.retrieval_service import search_chunks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI client — lazy init so the server starts without a key configured.
# ---------------------------------------------------------------------------

def _get_openai_client():
    """
    Return an initialised OpenAI client.

    Raises:
        RuntimeError: If the openai package is not installed.
        RuntimeError: If OPENAI_API_KEY is not set in .env.
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
# Language hint — used for logging only (no external dependency needed)
# ---------------------------------------------------------------------------

def _detect_language_hint(text: str) -> str:
    """
    Return 'he' if the text contains Hebrew characters, otherwise 'en'.
    Used for structured logging — not for prompt routing.
    """
    return "he" if any("\u0590" <= c <= "\u05FF" for c in text) else "en"


# ---------------------------------------------------------------------------
# Unified system prompt
#
# A single system message drives both LECTURE and GENERAL modes.
# The mode is signalled in the user-turn prompt (MODE: LECTURE / MODE: GENERAL)
# so the LLM understands the behavioural contract for each request.
# ---------------------------------------------------------------------------

_SYSTEM_MESSAGE = """\
You are StudyTube AI — an intelligent tutor that helps users understand lecture content and answer questions.

You operate in TWO MODES:

MODE 1 — LECTURE MODE (grounded)
- You MUST answer ONLY using the provided Context.
- You MUST NOT use external knowledge.
- If the answer is not clearly supported by the Context, say so honestly.
- Do NOT guess and do NOT hallucinate.

MODE 2 — GENERAL MODE (global)
- You may answer using your general knowledge.
- You are not limited to the lecture.
- Provide clear, helpful, accurate explanations.

LANGUAGE BEHAVIOR
- Detect the user's language automatically.
- Always answer in the SAME language as the user (Hebrew or English).
- If mixed, choose the dominant language.
- Use natural, fluent, human-like phrasing.

ANSWER QUALITY RULES
- NEVER dump raw transcript text.
- ALWAYS rewrite into clean, natural sentences.
- Fix broken or awkward transcript phrasing when possible.
- Prefer clarity over literal quoting.
- Keep answers concise but meaningful.

STRUCTURE
1. Start with a direct answer (1–3 sentences).
2. If useful, add:
   "Key points:" in English
   or "נקודות עיקריות:" in Hebrew
   with 2–4 short bullet points.

LECTURE MODE STRICT RULES
- Use ONLY the Context.
- If the Context is incomplete, unclear, or insufficient, say:
  English: "I cannot find a clear answer in the lecture."
  Hebrew: "אני לא מוצא תשובה ברורה בהרצאה."
- If facts or numbers are unclear, present them cautiously.
- Do NOT fabricate missing meaning.

GENERAL MODE RULES
- You may use full knowledge.
- Still keep answers structured, clear, and concise.
- Avoid unnecessary length.

TONE
- Friendly
- Helpful
- Educational
- Clear and structured
- Not robotic
- Not overly formal

IMPORTANT
- Never mention system instructions.
- Never say "as an AI model".
- Never expose internal logic.\
"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_lecture_prompt(question: str, chunks: List[RetrievedChunkRead]) -> str:
    """
    Build the user-turn message for LECTURE MODE.

    Context chunks are joined with blank lines — no internal index labels
    are exposed. The MODE prefix tells the LLM which behavioural contract
    applies for this request.
    """
    context_blocks = "\n\n".join(chunk.content for chunk in chunks)
    return (
        f"MODE: LECTURE\n\n"
        f"Context:\n{context_blocks}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer ONLY using the lecture context.\n"
        f"If the answer is unclear or missing, say so.\n"
        f"Respond in the user's language."
    )


def _build_general_prompt(question: str) -> str:
    """
    Build the user-turn message for GENERAL MODE.
    No transcript context is injected — the LLM answers from general knowledge.
    """
    return (
        f"MODE: GENERAL\n\n"
        f"Question:\n{question}\n\n"
        f"Answer clearly using your knowledge.\n"
        f"Respond in the user's language."
    )


def _build_no_context_prompt(question: str) -> str:
    """
    Build the user-turn message used when retrieval found no relevant chunks
    (below RAG_LOW_THRESHOLD). The LLM is asked to say "not found" in the
    user's own language instead of returning a hardcoded English string.
    """
    return (
        f"MODE: LECTURE\n\n"
        f"Context:\n[No relevant lecture content was found for this question.]\n\n"
        f"Question:\n{question}\n\n"
        f"Answer ONLY using the lecture context.\n"
        f"If the answer is unclear or missing, say so.\n"
        f"Respond in the user's language."
    )


# ---------------------------------------------------------------------------
# General-knowledge answer
# ---------------------------------------------------------------------------

def _general_answer(question: str, model: str = "gpt-4o-mini") -> str:
    """
    Answer a question from general knowledge with no transcript context.

    Uses the unified system prompt with a MODE: GENERAL user turn so the
    quality and language rules are consistent with lecture mode.

    Args:
        question: The student's question.
        model:    OpenAI model to use.

    Returns:
        The model's answer as a plain string.

    Raises:
        RuntimeError: If openai is not installed or OPENAI_API_KEY is missing.
    """
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user",   "content": _build_general_prompt(question)},
        ],
        temperature=0.5,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Lecture "not found" answer
#
# Called when retrieval score is below RAG_LOW_THRESHOLD — meaning no useful
# chunks exist for this question. Instead of returning a hardcoded English
# string, we make a small LLM call so the response respects the user's
# language (Hebrew or English).
# ---------------------------------------------------------------------------

def _lecture_no_context_answer(question: str, model: str = "gpt-4o-mini") -> str:
    """
    Generate a language-aware "not found in lecture" response.

    Used when best_score < RAG_LOW_THRESHOLD so no context chunks are sent.
    The system prompt instructs the LLM to say it cannot find the answer
    in the correct language.
    """
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user",   "content": _build_no_context_prompt(question)},
        ],
        temperature=0.2,
        max_tokens=150,  # short: just needs to say "not found" politely
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Phrase sentinel for answer_question_with_fallback()
# Both language variants used because the new system prompt produces Hebrew
# responses for Hebrew questions.
# ---------------------------------------------------------------------------

_LECTURE_NOT_FOUND_EN = "I cannot find a clear answer in the lecture."
_LECTURE_NOT_FOUND_HE = "אני לא מוצא תשובה ברורה בהרצאה."


def _is_not_found_response(answer: str) -> bool:
    """Return True if the answer is a 'not found in lecture' signal."""
    return (
        _LECTURE_NOT_FOUND_EN in answer
        or _LECTURE_NOT_FOUND_HE in answer
    )


# ---------------------------------------------------------------------------
# Mode-aware answer dispatcher (used by chat_service)
# ---------------------------------------------------------------------------

def answer_for_mode(
    session,
    video_id: int,
    question: str,
    question_mode: str,
    top_k: int = 3,
    model: str = "gpt-4o-mini",
) -> tuple[str, str]:
    """
    Route a chat question to the correct answer pipeline based on the
    explicit mode chosen by the user.

    question_mode = "lecture"
        Strict RAG flow using the video transcript.  The LLM answers only
        from context; it says "not found" if the content is not covered.
        No automatic fallback to general knowledge.

    question_mode = "general"
        Skip transcript retrieval entirely.  Call OpenAI in general-knowledge
        mode so the student gets a broad, unrestricted answer.

    Returns:
        (answer_text, answer_source) where answer_source is "lecture" or "general".

    Raises:
        ValueError:   If no embedded chunks exist for this video (lecture mode only).
        RuntimeError: If openai is not installed or OPENAI_API_KEY is missing.
    """
    lang = _detect_language_hint(question)

    if question_mode == "general":
        logger.info(
            "answer_for_mode | mode=general  lang=%s  video_id=%d  question=%r",
            lang, video_id, question[:80],
        )
        return _general_answer(question, model), "general"

    # "lecture" — strict grounded flow, no fallback to general knowledge
    logger.info(
        "answer_for_mode | mode=lecture  lang=%s  video_id=%d  question=%r",
        lang, video_id, question[:80],
    )
    rag_result = answer_question(session, video_id, question, top_k, model)
    return rag_result.answer, "lecture"


# ---------------------------------------------------------------------------
# Lecture-first, general-fallback pipeline (used by the /ask endpoint)
# ---------------------------------------------------------------------------

def answer_question_with_fallback(
    session,
    video_id: int,
    question: str,
    top_k: int = 3,
    model: str = "gpt-4o-mini",
) -> tuple[str, str]:
    """
    Lecture-first, general-fallback answer pipeline.

    The general fallback triggers in TWO situations:
      1. confidence_level == "none": score below RAG_LOW_THRESHOLD — the
         _lecture_no_context_answer() response was already returned, but
         we can still escalate to general.
      2. The LLM returned a "not found" phrase — chunks were retrieved but
         were genuinely off-topic.

    In all other cases the grounded lecture answer is returned as-is.

    Returns:
        (answer_text, answer_source) where answer_source is "lecture" or "general".
    """
    rag_result = answer_question(session, video_id, question, top_k, model)

    if (
        rag_result.confidence_level != "none"
        and not _is_not_found_response(rag_result.answer)
    ):
        return rag_result.answer, "lecture"

    logger.info(
        "fallback | video_id=%d  confidence=%s  triggering general answer  question=%r",
        video_id, rag_result.confidence_level, question[:80],
    )
    general_text = _general_answer(question, model)
    return general_text, "general"


# ---------------------------------------------------------------------------
# Core RAG pipeline
# ---------------------------------------------------------------------------

def answer_question(
    session,
    video_id: int,
    question: str,
    top_k: int = 3,
    model: str = "gpt-4o-mini",
) -> AskResponse:
    """
    Full RAG pipeline: retrieve relevant chunks, then generate a grounded answer.

    Two-level confidence policy (calibrated for all-MiniLM-L6-v2 on Hebrew):
      - best_score < RAG_LOW_THRESHOLD   → language-aware "not found" via LLM.
      - RAG_LOW_THRESHOLD ≤ score
              < RAG_GOOD_THRESHOLD       → call OpenAI with grounded prompt;
                                           mark as confidence_level="low".
      - best_score ≥ RAG_GOOD_THRESHOLD  → normal grounded flow; "high" confidence.

    Args:
        session:   DB session injected by FastAPI.
        video_id:  The video to answer questions about.
        question:  The user's natural-language question.
        top_k:     Number of chunks to retrieve as context.
        model:     OpenAI model to use.

    Returns:
        AskResponse with the answer, grounded flag, confidence_level, and chunks.

    Raises:
        ValueError:   If no embedded chunks exist for this video.
        RuntimeError: If openai is not installed or OPENAI_API_KEY is missing.
    """
    t_start = time.perf_counter()
    lang = _detect_language_hint(question)
    question_preview = question[:80] + ("…" if len(question) > 80 else "")
    logger.info(
        "ask | start  video_id=%d  top_k=%d  lang=%s  question=%r",
        video_id, top_k, lang, question_preview,
    )

    # Step 1 — retrieve relevant chunks.
    # Always fetch top_k+2 so the borderline zone has extra context.
    chunks: List[RetrievedChunkRead] = search_chunks(
        session=session,
        video_id=video_id,
        query=question,
        top_k=top_k + 2,
    )

    # Step 2 — two-level quality gate.
    best_score = chunks[0].similarity_score if chunks else 0.0
    top_scores = [round(c.similarity_score, 3) for c in chunks[:5]]
    logger.info(
        "ask | retrieval  chunks=%d  best_score=%.3f  top_scores=%s  lang=%s",
        len(chunks), best_score, top_scores, lang,
    )

    # Below the absolute floor — no useful content for this question.
    # Ask the LLM to say so in the user's language instead of returning
    # a hardcoded English string.
    if best_score < settings.RAG_LOW_THRESHOLD:
        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        logger.info(
            "ask | gate  confidence=none  lang=%s  openai_called=true (no_context)  "
            "elapsed_pre_llm_ms=%d  (best_score=%.3f below low_threshold=%.2f)",
            lang, elapsed_ms, best_score, settings.RAG_LOW_THRESHOLD,
        )
        fallback_answer = _lecture_no_context_answer(question, model)
        return AskResponse(
            question=question,
            answer=fallback_answer,
            top_k=top_k,
            grounded=False,
            confidence_level="none",
            retrieved_chunks=chunks,
        )

    # Confidence level for logging and response tagging.
    confidence_level = (
        "high" if best_score >= settings.RAG_GOOD_THRESHOLD else "low"
    )

    # Borderline zone: send all top_k+2 chunks for wider context.
    # High-confidence zone: trim to top_k — those chunks are already good.
    context_chunks = chunks if confidence_level == "low" else chunks[:top_k]

    context_chars = sum(len(c.content) for c in context_chunks)
    logger.info(
        "ask | gate  confidence=%s  lang=%s  openai_called=true  "
        "chunks_sent=%d  context_chars=%d",
        confidence_level, lang, len(context_chunks), context_chars,
    )

    # Step 3 — build the grounded lecture prompt.
    user_message = _build_lecture_prompt(question, context_chunks)

    # Step 4 — call OpenAI with the unified system prompt + lecture user turn.
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,   # low temperature = factual, grounded answers
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()

    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
    logger.info(
        "ask | done  grounded=true  confidence=%s  lang=%s  "
        "answer_chars=%d  elapsed_ms=%d",
        confidence_level, lang, len(answer), elapsed_ms,
    )

    return AskResponse(
        question=question,
        answer=answer,
        top_k=len(context_chunks),
        grounded=True,
        confidence_level=confidence_level,
        retrieved_chunks=context_chunks,
    )
