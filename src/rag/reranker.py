# src/rag/reranker.py
import time
from functools import lru_cache

from loguru import logger


@lru_cache(maxsize=1)
def _get_reranker():
    """
    Load the cross-encoder model once and cache it.

    lru_cache(maxsize=1) ensures the model is loaded only once
    across the application lifetime — loading takes ~2s the first time.

    Model choice: mmarco-mMiniLMv2-L12-H384-v1
    - Multilingual (covers French regulatory documents)
    - 12 layers, 384 hidden dims — good balance speed/quality
    - Trained on MS MARCO passage ranking dataset
    - ~120MB download, cached in ~/.cache/huggingface/

    Alternative for English-only: cross-encoder/ms-marco-MiniLM-L-6-v2
    (22MB, faster, but English only)
    """
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder model (first call only)...")
    model = CrossEncoder(
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        max_length=512,
    )
    logger.info("Cross-encoder loaded")
    return model


def rerank(
    question: str,
    chunks: list[str],
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """
    Re-score chunks against the question using a cross-encoder.

    The cross-encoder reads (question, chunk) pairs together,
    producing a relevance score that captures the precise relationship
    between the question and each chunk — unlike bi-encoders which
    encode them separately.

    This is the second stage of a two-stage retrieval pipeline:
      Stage 1 (fast)   : vector/hybrid search  → top 20-50 candidates
      Stage 2 (precise): cross-encoder rerank  → top 3-5 results

    Args:
        question: User question in natural language.
        chunks: Candidate chunks from stage-1 retrieval (plain text, no metadata).
        top_k: Return only the top_k highest-scoring chunks.
                If None, return all chunks sorted by score.

    Returns:
        List of (chunk_text, score) tuples, sorted by score descending.
        Scores are logits (unbounded) — higher = more relevant.
        Typical range: -10 (irrelevant) to +10 (highly relevant).
    """
    if not chunks:
        return []

    start = time.perf_counter()
    model = _get_reranker()

    # Build (question, chunk) pairs for the cross-encoder
    pairs = [(question, chunk) for chunk in chunks]
    scores = model.predict(pairs)

    # Pair each chunk with its score and sort descending
    ranked = sorted(
        zip(chunks, scores.tolist(), strict=True),
        key=lambda x: x[1],
        reverse=True,
    )

    if top_k is not None:
        ranked = ranked[:top_k]

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Rerank | chunks_in={} | chunks_out={} | best={:.3f} | worst={:.3f} | latency={:.0f}ms",
        len(chunks),
        len(ranked),
        ranked[0][1] if ranked else 0,
        ranked[-1][1] if ranked else 0,
        latency_ms,
    )

    return ranked


async def retrieve_and_rerank(
    question: str,
    top_k_retrieve: int = 10,
    top_k_rerank: int = 3,
    use_hybrid: bool = True,
    doc_type: str | None = "regulatory",
) -> list[str]:
    """
    Full two-stage pipeline: retrieval + reranking.

    Stage 1 — Retrieve more candidates than needed (top_k_retrieve).
    Stage 2 — Rerank with cross-encoder, keep only top_k_rerank.

    The ratio top_k_retrieve / top_k_rerank is the "rerank window".
    Standard practice: retrieve 10-20x more than you return.
    Example: retrieve 20, rerank to top 3.

    Args:
        question: User question.
        top_k_retrieve: Number of candidates from stage 1 (default 10).
        top_k_rerank: Final number of chunks to return (default 3).
        use_hybrid: Use hybrid search for stage 1 (True) or vector-only (False).
        doc_type: Filter by document type.

    Returns:
        List of formatted chunk strings (with source metadata), reranked.
    """
    from src.rag.hybrid_search import hybrid_search
    from src.rag.retrieval import retrieve_chunks

    # Stage 1 — retrieval
    if use_hybrid:
        candidates = await hybrid_search(
            question=question,
            top_k=top_k_retrieve,
            doc_type=doc_type,
        )
    else:
        candidates = await retrieve_chunks(
            question=question,
            top_k=top_k_retrieve,
            doc_type=doc_type,
            min_similarity=0.0,
        )

    if not candidates:
        return []

    # Split metadata prefix from text for reranking
    # Format is "[filename, Page X]\nchunk text"
    # Cross-encoder should score only the text content, not the metadata
    def _extract_text(formatted_chunk: str) -> str:
        lines = formatted_chunk.split("\n", 1)
        return lines[1] if len(lines) > 1 else lines[0]

    def _extract_metadata(formatted_chunk: str) -> str:
        return formatted_chunk.split("\n", 1)[0]

    texts = [_extract_text(c) for c in candidates]
    metadata = [_extract_metadata(c) for c in candidates]

    # Stage 2 — rerank on text only
    ranked = rerank(question, texts, top_k=top_k_rerank)

    # Rebuild formatted chunks with metadata in original order
    text_to_meta = {text: meta for text, meta in zip(texts, metadata, strict=True)}
    result = [f"{text_to_meta.get(chunk_text, '')}\n{chunk_text}" for chunk_text, _ in ranked]

    return result
