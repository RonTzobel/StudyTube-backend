from typing import List

from sqlmodel import Session

from app.config.settings import settings
from app.schemas.qa import AskResponse
from app.schemas.retrieval import RetrievedChunkRead
from app.services.retrieval_service import search_chunks


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

def answer_question(
    session,
    video_id: int,
    question: str,
    top_k: int = 3,
    model: str = "gpt-4o-mini",
) -> AskResponse:
    """
    Full RAG pipeline: retrieve relevant chunks, then generate an answer.

    Steps:
      1. Retrieve the top_k most semantically relevant transcript chunks
         for the question (reuses the existing retrieval_service).
      2. Build a prompt that injects those chunks as context.
      3. Send the prompt to OpenAI and return the grounded answer.

    This is what makes it RAG (Retrieval-Augmented Generation):
      - Pure LLM: answers from training data → may hallucinate.
      - RAG: answers from YOUR retrieved content → grounded in the transcript.

    Args:
        session:   DB session injected by FastAPI.
        video_id:  The video to answer questions about.
        question:  The user's natural-language question.
        top_k:     Number of chunks to retrieve as context.
        model:     OpenAI model to use. gpt-4o-mini is fast and cheap;
                   swap to gpt-4o for higher quality if needed.

    Returns:
        AskResponse with the answer and the retrieved chunks.

    Raises:
        ValueError:   If no embedded chunks exist for this video (from search_chunks).
        RuntimeError: If openai is not installed or OPENAI_API_KEY is missing.
    """
    # Step 1 — retrieve relevant chunks using the existing retrieval service
    chunks: List[RetrievedChunkRead] = search_chunks(
        session=session,
        video_id=video_id,
        query=question,
        top_k=top_k,
    )

    # Step 2 — build the prompt
    user_message = _build_prompt(question, chunks)

    # Step 3 — call OpenAI
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

    return AskResponse(
        question=question,
        answer=answer,
        top_k=top_k,
        retrieved_chunks=chunks,
    )
