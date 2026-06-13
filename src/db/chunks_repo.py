# src/db/chunks_repo.py
from dataclasses import dataclass

from loguru import logger

from src.db.database import get_connection


@dataclass
class ChunkRecord:
    """A chunk retrieved from pgvector with its similarity score."""

    id: int
    document_id: int
    chunk_index: int
    text: str
    token_count: int
    chunk_strategy: str
    page_number: int | None
    section_title: str | None
    similarity: float  # cosine similarity (1 - cosine distance)
    filename: str  # from joined documents table


async def insert_document(
    filename: str,
    doc_type: str = "regulatory",
    source_url: str | None = None,
) -> int:
    """
    Insert a document record and return its ID.

    Args:
        filename: Original filename (e.g., "rgpd_2016_679.pdf").
        doc_type: Document type for filtering.
        source_url: Optional URL source.

    Returns:
        The new document's ID.
    """
    async with get_connection() as conn:
        doc_id = await conn.fetchval(
            """
            INSERT INTO documents (filename, doc_type, source_url)
            VALUES ($1, $2, $3)
            RETURNING id
            """,
            filename,
            doc_type,
            source_url,
        )
    logger.info("Document inserted | id={} | filename={}", doc_id, filename)
    return doc_id


async def insert_chunks(
    document_id: int,
    chunks_data: list[dict],
) -> int:
    """
    Bulk insert chunks with their embeddings into pgvector.

    Uses executemany for efficiency — avoids N separate INSERT calls.

    Args:
        document_id: Parent document ID.
        chunks_data: List of dicts with keys:
            chunk_index, text, token_count, chunk_strategy,
            page_number, section_title, embedding (list[float]).

    Returns:
        Number of chunks inserted.
    """
    async with get_connection() as conn:
        await conn.executemany(
            """
            INSERT INTO chunks
                (document_id, chunk_index, text, token_count,
                 chunk_strategy, page_number, section_title, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector)
            """,
            [
                (
                    document_id,
                    c["chunk_index"],
                    c["text"],
                    c["token_count"],
                    c["chunk_strategy"],
                    c.get("page_number"),
                    c.get("section_title"),
                    str(c["embedding"]),  # asyncpg expects vector as string "[x,y,z]"
                )
                for c in chunks_data
            ],
        )

    logger.info(
        "Chunks inserted | document_id={} | count={}",
        document_id,
        len(chunks_data),
    )
    return len(chunks_data)


async def similarity_search(
    query_embedding: list[float],
    top_k: int = 5,
    doc_type: str | None = None,
    min_similarity: float = 0.0,
) -> list[ChunkRecord]:
    """
    Find the most similar chunks to a query embedding using HNSW index.

    Uses cosine distance (<=> operator). Returns chunks ordered by
    similarity descending (most relevant first).

    The HNSW index makes this O(log n) instead of O(n) brute force.

    Args:
        query_embedding: Embedding vector of the user question.
        top_k: Number of chunks to return.
        doc_type: Optional filter by document type (e.g., "regulatory").
        min_similarity: Minimum similarity threshold (0.0 = no filter).

    Returns:
        List of ChunkRecord ordered by similarity descending.
    """
    embedding_str = str(query_embedding)

    # Build query with optional doc_type filter
    # 1 - (cosine distance) = cosine similarity
    if doc_type:
        query = """
            SELECT
                c.id, c.document_id, c.chunk_index, c.text,
                c.token_count, c.chunk_strategy, c.page_number,
                c.section_title, d.filename,
                1 - (c.embedding <=> $1::vector) AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.doc_type = $3
              AND 1 - (c.embedding <=> $1::vector) >= $4
            ORDER BY c.embedding <=> $1::vector
            LIMIT $2
        """
        params = (embedding_str, top_k, doc_type, min_similarity)
    else:
        query = """
            SELECT
                c.id, c.document_id, c.chunk_index, c.text,
                c.token_count, c.chunk_strategy, c.page_number,
                c.section_title, d.filename,
                1 - (c.embedding <=> $1::vector) AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE 1 - (c.embedding <=> $1::vector) >= $3
            ORDER BY c.embedding <=> $1::vector
            LIMIT $2
        """
        params = (embedding_str, top_k, min_similarity)

    async with get_connection() as conn:
        rows = await conn.fetch(query, *params)

    results = [
        ChunkRecord(
            id=row["id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            token_count=row["token_count"],
            chunk_strategy=row["chunk_strategy"],
            page_number=row["page_number"],
            section_title=row["section_title"],
            similarity=float(row["similarity"]),
            filename=row["filename"],
        )
        for row in rows
    ]

    logger.debug(
        "Similarity search | top_k={} | results={} | best_score={:.4f}",
        top_k,
        len(results),
        results[0].similarity if results else 0.0,
    )

    return results
