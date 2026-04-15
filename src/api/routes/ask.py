# src/api/routes/ask.py
from fastapi import APIRouter, HTTPException
from loguru import logger

from src.api.schemas.ask import AskRequest, AskResponse, Citation
from src.prompts.rag_prompts import (
    RAG_STRUCTURED_SYSTEM_PROMPT,
    parse_structured_response,
)
from src.security.prompt_guard import (
    detect_direct_injection,
    sanitize_chunks,
    build_sandwiched_user_message,
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
    Answer a regulatory question using the RAG pipeline with injection guards.

    Pipeline:
    1. Detect direct injection in user question → HTTP 400
    2. Retrieve relevant chunks (stub → pgvector in Bloc 2)
    3. Sanitize chunks against indirect injection
    4. Build sandwiched prompt (context + reminder + question)
    5. Validate context window
    6. Call Mistral AI
    7. Parse JSON response
    8. Return typed AskResponse
    """
    # Defense 1 — direct injection detection
    if detect_direct_injection(request.question):
        logger.warning("Blocked injection attempt | question='{}'", request.question)
        raise HTTPException(
            status_code=400,
            detail="Your question contains patterns that cannot be processed.",
        )

    logger.info("RAG query | question='{}' | top_k={}", request.question, request.top_k)

    # Retrieve + sanitize
    raw_chunks = await _retrieve_chunks_stub(request.question, request.top_k)
    chunks = sanitize_chunks(raw_chunks)  # Defense 2 — indirect injection

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No valid document chunks available after sanitization.",
        )

    # Defense 3 — sandwich prompt structure
    user_message = build_sandwiched_user_message(
        question=request.question,
        context_chunks=chunks,
    )

    messages = [
        {"role": "system", "content": RAG_STRUCTURED_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

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