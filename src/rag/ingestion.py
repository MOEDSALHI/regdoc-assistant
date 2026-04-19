# src/rag/ingestion.py
import time
from pathlib import Path

from loguru import logger

from src.db.chunks_repo import insert_chunks, insert_document
from src.db.database import get_connection
from src.embeddings.chunker import Chunk, select_chunking_strategy
from src.embeddings.embedder import embed_batch


async def _document_exists(filename: str) -> int | None:
    """
    Check if a document with this filename already exists.

    Returns the existing document ID or None.
    Prevents duplicate ingestion on re-runs.
    """
    async with get_connection() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM documents WHERE filename = $1",
            filename,
        )
    return row["id"] if row else None


async def _delete_document(document_id: int) -> None:
    """
    Delete a document and all its chunks (cascade).
    Used for rollback on partial ingestion failure.
    """
    async with get_connection() as conn:
        await conn.execute(
            "DELETE FROM documents WHERE id = $1",
            document_id,
        )
    logger.warning("Document {} deleted (rollback)", document_id)


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract text from a PDF file, page by page.

    Returns a list of dicts with 'page_number' and 'text' keys.
    Pages with less than 50 characters are skipped (likely empty or images).

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of {'page_number': int, 'text': str} dicts.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required: uv add pypdf")

    reader = PdfReader(str(pdf_path))
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()

        if len(text) < 50:
            logger.debug("Skipping page {} (too short: {} chars)", i + 1, len(text))
            continue

        pages.append({"page_number": i + 1, "text": text})

    logger.info(
        "PDF extracted | file={} | total_pages={} | valid_pages={}",
        pdf_path.name,
        len(reader.pages),
        len(pages),
    )
    return pages


def extract_text_from_string(text: str) -> list[dict]:
    """
    Wrap a plain text string as a single-page document.
    Used for testing without real PDF files.
    """
    return [{"page_number": 1, "text": text}]


async def ingest_document(
    source: Path | str,
    doc_type: str = "regulatory",
    chunk_size: int = 512,
    overlap: int = 64,
    force_reingest: bool = False,
) -> dict:
    """
    Full ingestion pipeline: source → chunks → embeddings → pgvector.

    Steps:
    1. Check for existing document (idempotency guard)
    2. Extract text (PDF or plain text)
    3. Chunk text using the appropriate strategy
    4. Batch embed all chunks in one API call
    5. Insert document + chunks into pgvector
    6. Rollback on any failure

    Args:
        source: Path to PDF file, or plain text string for testing.
        doc_type: Document type tag ("regulatory", "guide", etc.).
        chunk_size: Target chunk size in tokens.
        overlap: Token overlap between consecutive chunks.
        force_reingest: If True, delete existing document and re-ingest.

    Returns:
        Dict with ingestion summary (doc_id, chunks_count, latency_ms).
    """
    start = time.perf_counter()

    # Determine filename and content
    if isinstance(source, Path):
        filename = source.name
        pages = extract_text_from_pdf(source)
    else:
        # Plain text input (for testing)
        filename = "inline_text.txt"
        pages = extract_text_from_string(source)

    # Step 1 — Idempotency check
    existing_id = await _document_exists(filename)
    if existing_id and not force_reingest:
        logger.info(
            "Document '{}' already ingested (id={}). Skipping. "
            "Use force_reingest=True to re-ingest.",
            filename, existing_id,
        )
        return {"doc_id": existing_id, "chunks_count": 0, "skipped": True}

    if existing_id and force_reingest:
        await _delete_document(existing_id)
        logger.info("Re-ingesting document '{}'", filename)

    # Step 2 — Chunk all pages
    all_chunks: list[Chunk] = []
    for page in pages:
        page_chunks = select_chunking_strategy(
            text=page["text"],
            document_type=doc_type,
            chunk_size=chunk_size,
            overlap=overlap,
            metadata={
                "filename": filename,
                "page_number": page["page_number"],
                "doc_type": doc_type,
            },
        )
        # Override page_number from PDF extraction
        for chunk in page_chunks:
            chunk.metadata["page_number"] = page["page_number"]
        all_chunks.extend(page_chunks)

    if not all_chunks:
        raise ValueError(f"No chunks extracted from '{filename}'. Check the source.")

    logger.info(
        "Chunking complete | file={} | chunks={} | avg_tokens={:.0f}",
        filename,
        len(all_chunks),
        sum(c.token_count for c in all_chunks) / len(all_chunks),
    )

    # Step 3 — Batch embed all chunks (single API call)
    texts = [chunk.text for chunk in all_chunks]
    embeddings = await embed_batch(texts)

    # Step 4 — Insert into pgvector (with rollback on failure)
    doc_id = await insert_document(filename, doc_type=doc_type)

    try:
        chunks_data = [
            {
                "chunk_index": chunk.index,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "chunk_strategy": chunk.metadata.get("chunk_strategy", "unknown"),
                "page_number": chunk.metadata.get("page_number"),
                "section_title": chunk.metadata.get("section_title"),
                "embedding": embedding,
            }
            for chunk, embedding in zip(all_chunks, embeddings)
        ]

        await insert_chunks(doc_id, chunks_data)

    except Exception as e:
        # Rollback: remove the partially inserted document
        logger.error("Ingestion failed after document insert — rolling back | error={}", e)
        await _delete_document(doc_id)
        raise

    latency_ms = (time.perf_counter() - start) * 1000

    summary = {
        "doc_id": doc_id,
        "filename": filename,
        "doc_type": doc_type,
        "chunks_count": len(all_chunks),
        "pages_processed": len(pages),
        "avg_tokens_per_chunk": sum(c.token_count for c in all_chunks) / len(all_chunks),
        "latency_ms": round(latency_ms),
        "skipped": False,
    }

    logger.info(
        "Ingestion complete | doc_id={} | file={} | chunks={} | latency={:.0f}ms",
        doc_id, filename, len(all_chunks), latency_ms,
    )

    return summary