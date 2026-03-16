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
# OpenAI client — created lazily on first use so the server starts even
# if the openai package is not installed or the key is not yet configured.
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
# Prompt builder
#
# The prompt is the most important part of a RAG system.
# Two rules keep answers grounded in the transcript:
#
#   1. The system message tells the LLM it ONLY has the provided context —
#      it must not use outside knowledge.
#   2. The user message explicitly asks for "I don't know" if the answer
#      is not in the context — this prevents confident hallucinations.
#
# The numbered-chunk format makes it easy to audit which chunk the answer
# came from during development.
# ---------------------------------------------------------------------------

def _build_prompt(question: str, chunks: List[RetrievedChunkRead]) -> str:
    """
    Build the user-turn message that combines context chunks with the question.

    The system message (sent separately in the API call) already tells the
    LLM to stay within the provided context. This function builds the
    user-turn content only.

    Args:
        question:  The user's question.
        chunks:    The retrieved transcript chunks to use as context.

    Returns:
        A formatted string combining the context and the question.
    """
    context_blocks = "\n\n".join(
        f"[Chunk {i + 1}]\n{chunk.content}"
        for i, chunk in enumerate(chunks)
    )

    return (
        f"Transcript context:\n"
        f"{context_blocks}\n\n"
        f"Question: {question}"
    )


_SYSTEM_MESSAGE = (
    "You are a helpful assistant for a video learning platform. "
    "You are given excerpts from a video transcript and a question about the video. "
    "Answer the question using ONLY the information in the provided transcript excerpts. "
    "If the answer cannot be found in the excerpts, say clearly: "
    "'I could not find the answer to that question in this video.' "
    "Do not use any knowledge outside the provided transcript. "
    "Answer in the same language as the question."
)


# ---------------------------------------------------------------------------
# Main QA function
# ---------------------------------------------------------------------------

_FALLBACK_ANSWER = (
    "I could not find enough relevant information in this video to answer confidently. "
    "Try rephrasing your question or asking about a topic that is covered in the transcript."
)


# ---------------------------------------------------------------------------
# General-knowledge fallback
#
# Used only when the lecture retrieval score is below RAG_LOW_THRESHOLD,
# meaning the transcript almost certainly does not cover the question.
# In that case, instead of returning the static fallback string, we call
# OpenAI in a general-assistant mode so the student still gets a useful answer.
#
# This is intentionally a separate prompt and separate function from the
# grounded lecture flow — the two modes must never be mixed.
# ---------------------------------------------------------------------------

_GENERAL_SYSTEM_MESSAGE = (
    "You are a helpful AI study assistant. "
    "Answer the student's question using your general knowledge. "
    "Be clear, concise, and educational. "
    "Answer in the same language as the question."
)


def _general_answer(question: str, model: str = "gpt-4o-mini") -> str:
    """
    Answer a question from general knowledge, with no transcript context.

    Called when the lecture retrieval score is below RAG_LOW_THRESHOLD,
    meaning the transcript does not cover the question at all.

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
            {"role": "system", "content": _GENERAL_SYSTEM_MESSAGE},
            {"role": "user",   "content": question},
        ],
        temperature=0.5,   # slightly higher than lecture mode — general answers are more open-ended
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# The exact phrase the strict system prompt instructs the LLM to use when it
# cannot find the answer in the provided transcript excerpts.
# Used to detect the "OpenAI was called but couldn't answer from the lecture"
# case — which must also fall back to general knowledge.
_LECTURE_NOT_FOUND_PHRASE = "I could not find the answer to that question in this video."


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
        Strict RAG flow using the video transcript. The LLM will say it
        cannot find the answer if the content is not covered. No automatic
        fallback to general knowledge — the user asked for lecture mode.

    question_mode = "general"
        Skip transcript retrieval entirely. Call OpenAI in general-knowledge
        mode directly, same as _general_answer().

    Returns:
        (answer_text, answer_source) where answer_source is "lecture" or "general".

    Raises:
        ValueError:   If no embedded chunks exist for this video (lecture mode only).
        RuntimeError: If openai is not installed or OPENAI_API_KEY is missing.
    """
    if question_mode == "general":
        logger.info(
            "answer_for_mode | general  video_id=%d  question=%r",
            video_id, question[:80],
        )
        return _general_answer(question, model), "general"

    # "lecture" — strict grounded flow, no fallback
    logger.info(
        "answer_for_mode | lecture  video_id=%d  question=%r",
        video_id, question[:80],
    )
    rag_result = answer_question(session, video_id, question, top_k, model)
    return rag_result.answer, "lecture"


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
      1. confidence_level == "none": score below RAG_LOW_THRESHOLD — no chunks
         were relevant enough to send to OpenAI at all.
      2. confidence_level == "low" or "high" BUT the LLM answered with the
         "I could not find the answer" phrase — meaning chunks were found but
         were genuinely off-topic (e.g. a borderline similarity score caused by
         language mismatch on an unrelated question).

    In all other cases the grounded lecture answer is returned as-is.

    Args:
        session:   DB session.
        video_id:  The video to query.
        question:  The student's question.
        top_k:     Number of chunks to retrieve as context.
        model:     OpenAI model to use.

    Returns:
        (answer_text, answer_source) where answer_source is "lecture" or "general".

    Raises:
        ValueError:   If no embedded chunks exist for this video (from answer_question).
        RuntimeError: If openai is not installed or OPENAI_API_KEY is missing.
    """
    rag_result = answer_question(session, video_id, question, top_k, model)

    if (
        rag_result.confidence_level != "none"
        and _LECTURE_NOT_FOUND_PHRASE not in rag_result.answer
    ):
        # Grounded answer: OpenAI was called with transcript chunks AND gave a
        # real answer. The lecture is the source of truth.
        return rag_result.answer, "lecture"

    # Fall through to general AI in two cases:
    #   - confidence_level == "none": no relevant chunks, OpenAI was not called.
    #   - OpenAI was called but explicitly said it couldn't find the answer
    #     (the LLM returned _LECTURE_NOT_FOUND_PHRASE).
    logger.info(
        "fallback | video_id=%d  confidence=%s  triggering general answer  question=%r",
        video_id, rag_result.confidence_level, question[:80],
    )
    general_text = _general_answer(question, model)
    return general_text, "general"


def answer_question(
    session,
    video_id: int,
    question: str,
    top_k: int = 3,
    model: str = "gpt-4o-mini",
) -> AskResponse:
    """
    Full RAG pipeline: retrieve relevant chunks, then generate an answer.

    Two-level confidence policy (calibrated for all-MiniLM-L6-v2 on Hebrew):
      - best_score < RAG_LOW_THRESHOLD   → immediate fallback, no OpenAI call.
      - RAG_LOW_THRESHOLD ≤ score
              < RAG_GOOD_THRESHOLD       → call OpenAI with the same strict
                                           grounded prompt; mark as "low" confidence.
      - best_score ≥ RAG_GOOD_THRESHOLD  → normal grounded flow, "high" confidence.

    Why two levels?
    all-MiniLM-L6-v2 is English-trained. For Hebrew content it still produces
    meaningful vectors, but scores are systematically 30–50 % lower than for
    English queries. A score of ~0.20 on Hebrew is equivalent to ~0.35 on
    English — genuinely relevant, worth sending to the LLM.

    Args:
        session:   DB session injected by FastAPI.
        video_id:  The video to answer questions about.
        question:  The user's natural-language question.
        top_k:     Number of chunks to retrieve as context.
        model:     OpenAI model to use. gpt-4o-mini is fast and cheap.

    Returns:
        AskResponse with the answer, grounded flag, confidence_level, and chunks.

    Raises:
        ValueError:   If no embedded chunks exist for this video.
        RuntimeError: If openai is not installed or OPENAI_API_KEY is missing.
    """
    t_start = time.perf_counter()
    question_preview = question[:80] + ("…" if len(question) > 80 else "")
    logger.info("ask | start  video_id=%d  top_k=%d  question=%r", video_id, top_k, question_preview)

    # Step 1 — retrieve relevant chunks using the existing retrieval service.
    # We always fetch top_k + 2 so we have spare chunks available for the
    # borderline zone (see Step 2 below). The tiny extra retrieval cost is
    # worth the improved context coverage for weak-match questions.
    chunks: List[RetrievedChunkRead] = search_chunks(
        session=session,
        video_id=video_id,
        query=question,
        top_k=top_k + 2,
    )

    # Step 2 — two-level quality gate.
    # chunks are returned highest-similarity-first, so index 0 is the best.
    best_score = chunks[0].similarity_score if chunks else 0.0
    top_scores = [round(c.similarity_score, 3) for c in chunks[:5]]
    logger.info("ask | retrieval  chunks=%d  best_score=%.3f  top_scores=%s", len(chunks), best_score, top_scores)

    # Below the absolute floor → content is almost certainly off-topic.
    # Save the API token; return fallback immediately.
    if best_score < settings.RAG_LOW_THRESHOLD:
        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        logger.info(
            "ask | gate  confidence=none  openai_called=false  elapsed_ms=%d  "
            "(best_score=%.3f below low_threshold=%.2f)",
            elapsed_ms, best_score, settings.RAG_LOW_THRESHOLD,
        )
        return AskResponse(
            question=question,
            answer=_FALLBACK_ANSWER,
            top_k=top_k,
            grounded=False,
            confidence_level="none",
            retrieved_chunks=chunks,
        )

    # Determine confidence level before calling OpenAI.
    # This is passed through to the response so callers can decide how much
    # to trust the answer (e.g. show a "low confidence" badge in the UI).
    confidence_level = (
        "high" if best_score >= settings.RAG_GOOD_THRESHOLD else "low"
    )

    # In the borderline zone individual chunks are weakly relevant, so we
    # send all top_k+2 chunks to give the LLM a wider window to find the
    # answer. In the high-confidence zone trim back to the requested top_k —
    # those chunks are already high-quality and we don't need extras.
    context_chunks = chunks if confidence_level == "low" else chunks[:top_k]

    context_chars = sum(len(c.content) for c in context_chunks)
    logger.info(
        "ask | gate  confidence=%s  openai_called=true  "
        "chunks_sent=%d  context_chars=%d",
        confidence_level, len(context_chunks), context_chars,
    )

    # Step 3 — build the prompt
    user_message = _build_prompt(question, context_chunks)

    # Step 4 — call OpenAI.
    # We use the same strict grounded system message for BOTH confidence
    # levels. The LLM is already instructed to say "I don't know" if the
    # context doesn't contain the answer, so the borderline ("low") case is
    # safe: if the chunks are genuinely irrelevant, the LLM will say so.
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,   # low temperature = more factual, less creative
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()

    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
    logger.info(
        "ask | done  grounded=true  confidence=%s  answer_chars=%d  elapsed_ms=%d",
        confidence_level, len(answer), elapsed_ms,
    )

    return AskResponse(
        question=question,
        answer=answer,
        top_k=len(context_chunks),   # reflect actual chunks sent to the LLM
        grounded=True,
        confidence_level=confidence_level,
        retrieved_chunks=context_chunks,
    )
