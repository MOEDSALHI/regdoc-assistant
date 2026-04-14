# src/api/routes/ask.py
from fastapi import APIRouter, HTTPException
from loguru import logger

from src.api.schemas.ask import AskRequest, AskResponse, Citation
from src.prompts.rag_prompts import (
    build_rag_structured_messages,
    parse_structured_response,
)
from src.services.llm_client import chat_complete
from src.services.token_counter import fits_in_context, log_context_breakdown

router = APIRouter(tags=["rag"])


async def _retrieve_chunks_stub(question: str, top_k: int) -> list[str]:
    """
    Stub retriever — returns hardcoded RGPD/CNIL chunks for any question.

    Will be replaced in Bloc 2 by real pgvector similarity search.
    Signature is intentionally identical to the future retrieve_chunks().
    """
    all_chunks = [
        "[CNIL recommendation 2021, Page 12] Server access logs must be kept "
        "for a minimum of 6 months and a maximum of 1 year.",

        "[GDPR Article 5(1)(e), Page 4] Personal data shall be kept for no "
        "longer than necessary for the purposes for which it is processed.",

        "[CNIL recommendation 2021, Page 8] Application logs should be kept "
        "for a maximum of 6 months. Debug logs deleted when no longer needed.",

        "[GDPR Article 32, Page 18] Controllers shall implement appropriate "
        "technical measures including pseudonymisation and encryption.",

        "[ANSSI Guide 2023, Page 34] Security logs should be kept at least "
        "1 year to allow investigation of incidents detected late.",
    ]
    return all_chunks[:top_k]


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """
    Answer a regulatory question using the RAG pipeline.

    Pipeline:
    1. Retrieve relevant chunks (stub → pgvector in Bloc 2)
    2. Validate context window
    3. Build structured prompt
    4. Call Mistral AI
    5. Parse JSON response
    6. Return typed AskResponse with citations
    """
    logger.info("RAG query | question='{}' | top_k={}", request.question, request.top_k)

    chunks = await _retrieve_chunks_stub(request.question, request.top_k)

    messages = build_rag_structured_messages(
        question=request.question,
        context_chunks=chunks,
    )

    if not fits_in_context(messages):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Retrieved context exceeds context window. "
                f"Reduce top_k (currently {request.top_k})."
            ),
        )

    log_context_breakdown(
        system_prompt=messages[0]["content"],
        chunks=chunks,
        question=request.question,
    )

    raw_response = await chat_complete(
        messages=messages,
        temperature=request.temperature,
    )

    parsed = parse_structured_response(raw_response)

    if "_parse_error" in parsed:
        logger.error(
            "JSON parse failed | error={} | raw={}",
            parsed["_parse_error"],
            parsed.get("_raw_response", ""),
        )
        raise HTTPException(
            status_code=502,
            detail="Model returned unparseable response. Please retry.",
        )

    return AskResponse(
        answer=parsed.get("answer"),
        confidence=parsed.get("confidence", "LOW"),
        citations=[Citation(**c) for c in parsed.get("citations", [])],
        cannot_answer=parsed.get("cannot_answer", False),
        chunks_used=len(chunks),
        question=request.question,
    )