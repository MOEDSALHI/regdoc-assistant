# src/rag/hybrid_search.py
import time
from dataclasses import dataclass

from loguru import logger
from rank_bm25 import BM25Okapi

from src.db.chunks_repo import ChunkRecord, similarity_search
from src.db.database import get_connection
from src.embeddings.embedder import embed_text


@dataclass
class RankedChunk:
    """
    A chunk with its hybrid RRF score and source ranks for debugging.

    Keeping vector_rank and bm25_rank visible is intentional —
    it lets us audit why a chunk was ranked where it was.
    """
    text: str
    filename: str
    page_number: int | None
    section_title: str | None
    rrf_score: float
    vector_rank: int | None   # None if not in vector results
    bm25_rank: int | None     # None if not in BM25 results
    vector_score: float | None


async def _fetch_all_chunks(doc_type: str | None = None) -> list[ChunkRecord]:
    """
    Fetch all chunks from pgvector for BM25 corpus indexing.

    BM25 needs the full corpus to compute IDF (how rare is each term
    across ALL documents). This is the fundamental difference with
    vector search which works on a single query vs stored vectors.

    Scalability note: for >50k chunks, replace with PostgreSQL full-text
    search (tsvector) or Elasticsearch to avoid loading everything in RAM.
    """
    if doc_type:
        query = """
            SELECT c.id, c.document_id, c.chunk_index, c.text,
                   c.token_count, c.chunk_strategy, c.page_number,
                   c.section_title, d.filename, 0.0 AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.doc_type = $1
            ORDER BY c.id
        """
        params = (doc_type,)
    else:
        query = """
            SELECT c.id, c.document_id, c.chunk_index, c.text,
                   c.token_count, c.chunk_strategy, c.page_number,
                   c.section_title, d.filename, 0.0 AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.id
        """
        params = ()

    async with get_connection() as conn:
        rows = await conn.fetch(query, *params)

    return [
        ChunkRecord(
            id=row["id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            token_count=row["token_count"],
            chunk_strategy=row["chunk_strategy"],
            page_number=row["page_number"],
            section_title=row["section_title"],
            similarity=0.0,
            filename=row["filename"],
        )
        for row in rows
    ]


def _bm25_search(
    query: str,
    corpus: list[ChunkRecord],
    top_k: int,
) -> list[tuple[ChunkRecord, float]]:
    """
    Run BM25 search on an in-memory corpus.

    Tokenization: simple whitespace split, lowercased.
    For better results on French text, consider stemming (Snowball)
    or lemmatization — not implemented here to keep dependencies minimal.

    Returns top_k (chunk, score) pairs sorted by score descending.
    Chunks with score 0 (no term overlap) are excluded.
    """
    tokenized_corpus = [chunk.text.lower().split() for chunk in corpus]
    tokenized_query = query.lower().split()

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    # Pair each chunk with its score, filter zeros, sort descending
    scored = [
        (chunk, float(score))
        for chunk, score in zip(corpus, scores)
        if score > 0
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


def _reciprocal_rank_fusion(
    vector_results: list[ChunkRecord],
    bm25_results: list[tuple[ChunkRecord, float]],
    top_k: int,
    k: int = 60,
) -> list[RankedChunk]:
    """
    Merge vector and BM25 results with Reciprocal Rank Fusion.

    Key insight: RRF is score-agnostic — it only uses rank positions.
    This makes it robust when combining lists with incompatible score scales.

    A document appearing in BOTH lists gets contributions from both,
    naturally rewarding documents that are relevant on two different axes.

    k=60 is the industry standard from Cormack et al. (2009).
    Higher k = less penalty for lower ranks = more conservative fusion.
    """
    # Build rank maps: chunk_id -> rank (1-indexed)
    vector_ranks = {r.id: rank + 1 for rank, r in enumerate(vector_results)}
    bm25_ranks   = {r.id: rank + 1 for rank, (r, _) in enumerate(bm25_results)}

    # Lookup maps for chunk objects
    vector_map = {r.id: r for r in vector_results}
    bm25_map   = {r.id: r for r, _ in bm25_results}
    all_chunks = {**bm25_map, **vector_map}  # vector_map wins on conflict

    all_ids = set(vector_ranks) | set(bm25_ranks)

    # Compute RRF score for each unique document
    rrf_scores: dict[int, float] = {}
    for chunk_id in all_ids:
        score = 0.0
        if chunk_id in vector_ranks:
            score += 1.0 / (k + vector_ranks[chunk_id])
        if chunk_id in bm25_ranks:
            score += 1.0 / (k + bm25_ranks[chunk_id])
        rrf_scores[chunk_id] = score

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    results = []
    for chunk_id in sorted_ids[:top_k]:
        chunk = all_chunks[chunk_id]
        results.append(RankedChunk(
            text=chunk.text,
            filename=chunk.filename,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            rrf_score=rrf_scores[chunk_id],
            vector_rank=vector_ranks.get(chunk_id),
            bm25_rank=bm25_ranks.get(chunk_id),
            vector_score=chunk.similarity if chunk_id in vector_map else None,
        ))

    return results


async def hybrid_search(
    question: str,
    top_k: int = 5,
    doc_type: str | None = "regulatory",
    min_vector_similarity: float = 0.0,
    rrf_k: int = 60,
) -> list[str]:
    """
    Hybrid retrieval: vector similarity + BM25 full-text, merged with RRF.

    When to use hybrid vs vector-only:
    - Queries with exact regulatory terms (DPIA, RSSI, Article 17 bis)
    - Short queries where embedding signal is weak
    - When context_recall is low on vector-only (detected via RAGAS)

    When vector-only is sufficient:
    - Long natural language questions
    - Paraphrase queries ("combien de temps" vs "durée de conservation")

    Args:
        question: User question in natural language.
        top_k: Final number of chunks to return.
        doc_type: Filter by document type (None = all).
        min_vector_similarity: Pre-filter for vector results.
        rrf_k: RRF constant (default 60).

    Returns:
        Formatted chunk strings ready for prompt injection.
    """
    start = time.perf_counter()

    # Step 1 — vector search: fetch top_k*2 candidates for RRF
    query_embedding = await embed_text(question)
    vector_results = await similarity_search(
        query_embedding=query_embedding,
        top_k=top_k * 2,
        doc_type=doc_type,
        min_similarity=min_vector_similarity,
    )

    # Step 2 — BM25: fetch full corpus and search
    all_chunks = await _fetch_all_chunks(doc_type=doc_type)

    if not all_chunks:
        logger.warning("Empty corpus — falling back to vector-only")
        return [c.text for c in vector_results[:top_k]]

    bm25_results = _bm25_search(question, all_chunks, top_k=top_k * 2)

    # Step 3 — RRF fusion
    ranked = _reciprocal_rank_fusion(
        vector_results=vector_results,
        bm25_results=bm25_results,
        top_k=top_k,
        k=rrf_k,
    )

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Hybrid search | q='{}' | vector={} | bm25={} | final={} | {:.0f}ms",
        question[:50], len(vector_results), len(bm25_results),
        len(ranked), latency_ms,
    )

    # Log rank breakdown for debugging
    for i, r in enumerate(ranked):
        logger.debug(
            "  #{} rrf={:.4f} | vec_rank={} bm25_rank={} | {}",
            i + 1, r.rrf_score, r.vector_rank, r.bm25_rank, r.text[:60],
        )

    # Step 4 — format with source metadata
    formatted = []
    for chunk in ranked:
        source = f"[{chunk.filename}"
        if chunk.page_number:
            source += f", Page {chunk.page_number}"
        if chunk.section_title:
            source += f", {chunk.section_title}"
        source += "]"
        formatted.append(f"{source}\n{chunk.text}")

    return formatted