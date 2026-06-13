# src/api/routes/ask.py
import time

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.api.schemas.ask import AskRequest, AskResponse, Citation
from src.config import settings
from src.observability.metrics import (
    RAG_CHUNKS_RETRIEVED,
    RAG_INJECTION_BLOCKED,
    RAG_LLM_ERRORS,
    RAG_QUERIES_TOTAL,
    RAG_RETRIEVAL_DURATION,
)
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

from src.rag.retrieval import retrieve_chunks
from src.rag.hybrid_search import hybrid_search
from src.rag.reranker import retrieve_and_rerank
from src.rag.query_expansion import hyde_retrieve

router = APIRouter(tags=["rag"])


def _classify_llm_error(exc: Exception) -> str:
    """Map an exception to a Prometheus-friendly error_type label.

    Keep the cardinality bounded (Prometheus best practice).
    Avoid putting raw exception messages as labels.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return "rate_limit"
    if status_code in (502, 503, 504):
        return "upstream"
    if isinstance(exc, TimeoutError) or "timeout" in str(exc).lower():
        return "timeout"
    return "other"
# async def _retrieve_chunks_stub(question: str, top_k: int) -> list[str]:
#     """
#     Stub retriever — returns hardcoded RGPD/CNIL chunks for any question.

#     Will be replaced in Bloc 2 by real pgvector similarity search.
#     Signature is intentionally identical to the future retrieve_chunks().
#     """
#     all_chunks = [
#         "[CNIL recommendation 2021, Page 12] Server access logs must be kept "
#         "for a minimum of 6 months and a maximum of 1 year.",

#         "[GDPR Article 5(1)(e), Page 4] Personal data shall be kept for no "
#         "longer than necessary for the purposes for which it is processed.",

#         "[CNIL recommendation 2021, Page 8] Application logs should be kept "
#         "for a maximum of 6 months. Debug logs deleted when no longer needed.",

#         "[GDPR Article 32, Page 18] Controllers shall implement appropriate "
#         "technical measures including pseudonymisation and encryption.",

#         "[ANSSI Guide 2023, Page 34] Security logs should be kept at least "
#         "1 year to allow investigation of incidents detected late.",
#     ]
#     return all_chunks[:top_k]


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """
    Answer a regulatory question using the RAG pipeline.

    Pipeline (instrumented with Prometheus metrics):
    1. Detect direct injection -> HTTP 400 (counted: rag_injection_blocked_total)
    2. Retrieve real chunks from pgvector (timed: rag_retrieval_duration_seconds)
    3. Sanitize chunks against indirect injection
    4. Build sandwiched prompt
    5. Validate context window
    6. Call Mistral AI (timed inside llm_client)
    7. Parse JSON response
    8. Return typed AskResponse (counted: rag_queries_total)
    """
    # Defense 1 — direct injection detection (NOT counted as a RAG query)
    if detect_direct_injection(request.question):
        logger.warning("Blocked injection | question='{}'", request.question)
        RAG_INJECTION_BLOCKED.inc()
        raise HTTPException(
            status_code=400,
            detail="Your question contains patterns that cannot be processed.",
        )

    # Real retrieval from pgvector
    mode = request.retrieval_mode   # MOVED UP (now used in metrics labels)
    logger.info(
        "RAG query | question='{}' | top_k={} | mode={}",
        request.question, request.top_k, mode,
    )

    try:
        # --- Retrieval phase (instrumented) ---
        t_retrieval = time.perf_counter()

        if mode == "naive":
            raw_chunks = await retrieve_chunks(
                question=request.question,
                top_k=request.top_k,
                doc_type="regulatory",
                min_similarity=0.70,
            )
        elif mode == "hybrid":
            raw_chunks = await hybrid_search(
                question=request.question,
                top_k=request.top_k,
                doc_type="regulatory",
            )
        elif mode == "reranked":
            raw_chunks = await retrieve_and_rerank(
                question=request.question,
                # top_k=request.top_k,
                top_k_rerank=request.top_k,
                doc_type="regulatory",
            )
        elif mode == "hyde":
            raw_chunks = await hyde_retrieve(
                question=request.question,
                top_k=request.top_k,
                doc_type="regulatory",
            )

        # observe retrieval metrics
        RAG_RETRIEVAL_DURATION.labels(mode=mode).observe(
            time.perf_counter() - t_retrieval
        )
        RAG_CHUNKS_RETRIEVED.labels(mode=mode).observe(len(raw_chunks))

        chunks = sanitize_chunks(raw_chunks)  # Defense 2 — indirect injection


        if not chunks:
            # No relevant chunks found — return cannot_answer directly
            RAG_QUERIES_TOTAL.labels(mode=mode, status="success").inc()
            return AskResponse(
                answer=None,
                confidence="LOW",
                citations=[],
                cannot_answer=True,
                chunks_used=0,
                question=request.question,
            )

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
                detail=f"Context exceeds window. Reduce top_k (currently {request.top_k}).",
            )

        log_context_breakdown(
            system_prompt=messages[0]["content"],
            chunks=chunks,
            question=request.question,
        )

        # LLM call — duration and tokens recorded inside chat_complete
        raw_response = await chat_complete(
            messages=messages,
            temperature=request.temperature,
        )

        parsed = parse_structured_response(raw_response)

        if "_parse_error" in parsed:
            logger.error("JSON parse failed | error={}", parsed["_parse_error"])
            raise HTTPException(status_code=502, detail="Model returned unparseable response.")

        RAG_QUERIES_TOTAL.labels(mode=mode, status="success").inc()
        return AskResponse(
            answer=parsed.get("answer"),
            confidence=parsed.get("confidence", "LOW"),
            citations=[Citation(**c) for c in parsed.get("citations", [])],
            cannot_answer=parsed.get("cannot_answer", False),
            chunks_used=len(chunks),
            question=request.question,
        )

    except HTTPException:
        # 4xx/5xx raised intentionally — count as error, don't re-classify
        RAG_QUERIES_TOTAL.labels(mode=mode, status="error").inc()
        raise
    except Exception as exc:
        # Unexpected — likely LLM SDK error after retries exhausted
        RAG_LLM_ERRORS.labels(
            model=settings.mistral_model,
            error_type=_classify_llm_error(exc),
        ).inc()
        RAG_QUERIES_TOTAL.labels(mode=mode, status="error").inc()
        raise