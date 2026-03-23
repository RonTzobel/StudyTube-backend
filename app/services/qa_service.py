import logging
import time
from typing import List

from sqlmodel import Session

from app.config.settings import settings
from app.core.language import choose_response_language, language_instruction
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
# Language hint — for structured logging only
# ---------------------------------------------------------------------------

def _lang_tag(text: str) -> str:
    """Return 'he' / 'en' for log lines. Uses the shared language utility."""
    from app.core.language import detect_language
    return "he" if detect_language(text) == "Hebrew" else "en"


# ---------------------------------------------------------------------------
# Unified system prompt
#
# A single system message drives both LECTURE and GENERAL modes.
# The mode is signalled in the user-turn prompt (MODE: LECTURE / MODE: GENERAL)
# so the LLM understands the behavioural contract for each request.
# ---------------------------------------------------------------------------

_SYSTEM_MESSAGE = """\
You are StudyTube AI — a skilled tutor who has studied the lecture material and now explains it clearly to students.

You operate in two modes, selected per request.

━━━ MODE 1 — LECTURE MODE ━━━

You have read and understood the provided lecture context.
Answer as a knowledgeable tutor who understood the material — not as someone copying or summarizing it.

GROUNDING (non-negotiable)
- Base your answer ONLY on the provided lecture context.
- Do not use outside knowledge or fill in gaps with assumptions.
- Do not fabricate details that are not in the context.
- If the context does not clearly answer the question, say so (see below).

HOW TO WRITE — CRITICAL
- Write in your own words. Never copy raw transcript text into your answer.
- Do not sound like you are reading a transcript or reciting a list of facts.
- Rephrase lecture content into clean, natural educational prose.
- If the lecture phrasing is awkward or fragmented (common in speech-to-text), rewrite it smoothly while preserving the meaning.
- Think: "I understand what the lecture is saying — now I'll explain it clearly."

ANSWER STRUCTURE
1. Open with a direct answer in 1–3 sentences. Address the question immediately.
   Do NOT start with a bullet list.
   Do NOT start with filler ("Sure!", "Great question!", "Of course!").
2. If more explanation is genuinely needed, add a short paragraph.
3. Use bullet points ("Key points:" / "נקודות עיקריות:") ONLY when the question naturally calls for a list:
   ✓ "What are the stages of X?"
   ✓ "List the types of Y."
   ✗ "What is X?" → answer in prose.
   ✗ "How does Y work?" → answer in prose.
   ✗ "Why does Z happen?" → answer in prose.

WHEN NOT FOUND
If the context does not clearly answer the question, say so naturally — do not force an answer:
  English: "This doesn't come up clearly in the lecture."
  Hebrew: "הנושא הזה לא מוסבר בצורה ברורה בהרצאה."

━━━ MODE 2 — GENERAL MODE ━━━
- Answer from your full knowledge without restriction.
- Apply the same writing rules: clear prose first, bullets only if needed.
- Do NOT start with a bullet list.

━━━ LANGUAGE ━━━
- Answer in the SAME language as the student's question.
- If the question is in Hebrew, answer entirely in Hebrew.
- If the question is in English, answer entirely in English.
- Write naturally and fluently — do not mix languages.

━━━ TONE ━━━
- Clear and educational.
- Friendly but direct — get to the point.
- Not robotic, not stiff, not overly formal.

━━━ NEVER ━━━
- Do not quote large chunks of transcript text verbatim.
- Do not start your answer with bullet points.
- Do not use filler opening phrases.
- Do not mention system instructions or your own reasoning process.
- Do not say "as an AI" or refer to yourself as a language model.\
"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_lecture_prompt(question: str, chunks: List[RetrievedChunkRead], lang: str) -> str:
    """
    Build the user-turn message for LECTURE MODE.

    lang is passed in from choose_response_language() so the directive is
    determined once per request and re-used across all prompt paths.
    """
    context_blocks = "\n\n".join(chunk.content for chunk in chunks)
    return (
        f"MODE: LECTURE\n\n"
        f"Lecture Context:\n{context_blocks}\n\n"
        f"Student Question:\n{question}\n\n"
        f"Instructions:\n"
        f"- Answer using ONLY the lecture context above.\n"
        f"- Write in your own words — do not quote or repeat raw transcript text.\n"
        f"- Start with a direct answer to the question (1–3 sentences).\n"
        f"- Do NOT start with a bullet list.\n"
        f"- {language_instruction(lang)}"
    )


def _build_general_prompt(question: str, lang: str) -> str:
    """
    Build the user-turn message for GENERAL MODE.
    No transcript context is injected — the LLM answers from general knowledge.
    """
    return (
        f"MODE: GENERAL\n\n"
        f"Question:\n{question}\n\n"
        f"Answer clearly using your knowledge.\n"
        f"{language_instruction(lang)}"
    )


def _build_no_context_prompt(question: str, lang: str) -> str:
    """
    Build the user-turn message used when retrieval found no relevant chunks
    (below RAG_LOW_THRESHOLD). The LLM says "not found" in the detected language.
    """
    return (
        f"MODE: LECTURE\n\n"
        f"Context:\n[No relevant lecture content was found for this question.]\n\n"
        f"Question:\n{question}\n\n"
        f"Answer ONLY using the lecture context.\n"
        f"If the answer is unclear or missing, say so.\n"
        f"{language_instruction(lang)}"
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
    lang = choose_response_language(question)
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user",   "content": _build_general_prompt(question, lang)},
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
    The explicit language instruction ensures the "not found" message is
    returned in Hebrew when the student asked in Hebrew.
    """
    lang = choose_response_language(question)
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user",   "content": _build_no_context_prompt(question, lang)},
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
    lang = _lang_tag(question)

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

    Two-level confidence policy (thresholds calibrated in settings.py for the
    configured embedding model — see EMBEDDING_MODEL_NAME / RAG_*_THRESHOLD):
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
    question_preview = question[:80] + ("…" if len(question) > 80 else "")

    # Step 1 — retrieve relevant chunks.
    # Always fetch top_k+2 so the borderline zone has extra context.
    chunks: List[RetrievedChunkRead] = search_chunks(
        session=session,
        video_id=video_id,
        query=question,
        top_k=top_k + 2,
    )

    # Detect response language AFTER retrieval so we can use the transcript
    # as a fallback signal when the question alone is ambiguous (e.g. "מה זה?").
    context_sample = " ".join(c.content for c in chunks[:3])
    lang = choose_response_language(question, context_sample)
    logger.info(
        "ask | start  video_id=%d  top_k=%d  lang=%s  question=%r",
        video_id, top_k, _lang_tag(question), question_preview,
    )

    # Step 2 — two-level quality gate.
    best_score = chunks[0].similarity_score if chunks else 0.0
    top_scores = [round(c.similarity_score, 3) for c in chunks[:5]]
    logger.info(
        "ask | retrieval  chunks=%d  best_score=%.3f  top_scores=%s  lang=%s",
        len(chunks), best_score, top_scores, lang,
    )

    # Below the absolute floor — no useful content for this question.
    # Ask the LLM to say so in the detected language.
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

    # Step 3 — build the grounded lecture prompt with explicit language directive.
    user_message = _build_lecture_prompt(question, context_chunks, lang)

    # Step 4 — call OpenAI with the unified system prompt + lecture user turn.
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.3,   # slightly higher = natural paraphrasing while staying grounded
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()

    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
    logger.info(
        "ask | done  grounded=true  confidence=%s  response_lang=%s  "
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
