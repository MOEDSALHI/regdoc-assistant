# src/rag/retrieval.py
import time

from loguru import logger

from src.db.chunks_repo import ChunkRecord, similarity_search
from src.embeddings.embedder import embed_text


async def retrieve_chunks(
    question: str,
    top_k: int = 5,
    doc_type: str | None = "regulatory",
    min_similarity: float = 0.70,
) -> list[str]:
    """
    Retrieve the most relevant document chunks for a question.

    Full retrieval pipeline:
    1. Embed the question with mistral-embed
    2. Run cosine similarity search in pgvector (HNSW index)
    3. Filter by minimum similarity threshold
    4. Format chunks with source metadata for LLM injection

    This function replaces _retrieve_chunks_stub() in ask.py.
    The return type is intentionally identical to the stub
    (list[str]) — no changes needed in the endpoint.

    Args:
        question: User question in natural language.
        top_k: Maximum number of chunks to return.
        doc_type: Filter by document type (None = search all).
        min_similarity: Minimum cosine similarity (0.0-1.0).
                       Chunks below this threshold are discarded.

    Returns:
        List of formatted chunk strings ready for prompt injection.
        Empty list if no relevant chunks found (triggers cannot_answer).
    """
    start = time.perf_counter()

    # Step 1 — embed the question
    query_embedding = await embed_text(question)

    # Step 2 — similarity search in pgvector
    records: list[ChunkRecord] = await similarity_search(
        query_embedding=query_embedding,
        top_k=top_k,
        doc_type=doc_type,
        min_similarity=min_similarity,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    if not records:
        logger.warning(
            "No chunks found | question='{}' | min_similarity={} | top_k={}",
            question[:60],
            min_similarity,
            top_k,
        )
        return []

    # Step 3 — format chunks with source metadata
    # The format must match what the LLM expects in the RAG prompt
    formatted = []
    for record in records:
        source_info = f"[{record.filename}"
        if record.page_number:
            source_info += f", Page {record.page_number}"
        if record.section_title:
            source_info += f", {record.section_title}"
        source_info += "]"

        chunk_text = f"{source_info}\n{record.text}"
        formatted.append(chunk_text)

    logger.info(
        "Retrieval complete | question='{}' | results={} | "
        "best={:.4f} | worst={:.4f} | latency={:.0f}ms",
        question[:60],
        len(records),
        records[0].similarity,
        records[-1].similarity,
        latency_ms,
    )

    return formatted


async def retrieve_chunks_with_scores(
    question: str,
    top_k: int = 5,
    doc_type: str | None = "regulatory",
    min_similarity: float = 0.70,
) -> list[tuple[str, float]]:
    """
    Same as retrieve_chunks() but returns (chunk_text, similarity_score) tuples.

    Used for debugging, evaluation (RAGAS), and monitoring dashboards.
    Not used in the main /ask pipeline — only for inspection.
    """
    query_embedding = await embed_text(question)
    records = await similarity_search(
        query_embedding=query_embedding,
        top_k=top_k,
        doc_type=doc_type,
        min_similarity=min_similarity,
    )
    return [(r.text, r.similarity) for r in records]
