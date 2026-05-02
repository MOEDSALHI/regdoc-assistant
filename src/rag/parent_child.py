# src/rag/parent_child.py
import re
import time
from loguru import logger

from src.db.database import get_connection
from src.embeddings.chunker import Chunk, chunk_by_article, chunk_fixed_size
from src.embeddings.embedder import embed_batch
from src.services.token_counter import count_tokens


async def ingest_parent_child(
    text: str,
    document_id: int,
    parent_chunk_size: int = 512,
    child_chunk_size: int = 80,
    doc_type: str = "regulatory",
    page_number: int = 1,
) -> dict:
    """
    Ingest text using parent-child chunking strategy.

    Parents: full article sections — stored as-is for LLM context, no embedding.
    Children: small fixed-size sub-chunks — embedded for precise retrieval.

    Key fix vs naive implementation: parents are NOT chunked by token limit.
    We use the natural article boundaries as parents, even if they exceed
    parent_chunk_size. This preserves the semantic integrity of each article.

    Args:
        text: Document text to ingest.
        document_id: Parent document ID in the documents table.
        parent_chunk_size: Token size for parent chunks (context for LLM).
        child_chunk_size: Token size for child chunks (retrieval precision).
        doc_type: Document type tag.
        page_number: Page number for metadata.

    Returns:
        Dict with counts of parents and children inserted.
    """

    start = time.perf_counter()

    # Step 1 — split into parent sections using article markers
    # Each "Article X" becomes one parent — we do NOT enforce a token limit here
    article_pattern = re.compile(
        r"(?:^|\n)(?:Article|Art\.?)\s+\d+",
        re.MULTILINE,
    )
    matches = list(article_pattern.finditer(text))

    if matches:
        parent_texts = []
        for i, match in enumerate(matches):
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section = text[start_pos:end_pos].strip()
            if section:
                parent_texts.append(section)
    else:
        # No article markers — use the full text as a single parent
        parent_texts = [text.strip()]

    if not parent_texts:
        return {"parents": 0, "children": 0}

    # Step 2 — create children from each parent using fixed-size chunking
    all_children: list[tuple[str, int]] = []  # (child_text, parent_index)

    for parent_idx, parent_text in enumerate(parent_texts):
        child_chunks = chunk_fixed_size(
            parent_text,
            chunk_size=child_chunk_size,
            overlap=10,
            metadata={"chunk_strategy": "child"},
        )
        for child in child_chunks:
            if len(child.text.split()) >= 5:  # skip micro-fragments < 5 words
                all_children.append((child.text, parent_idx))

    if not all_children:
        logger.warning("No valid children generated for document_id={}", document_id)
        return {"parents": 0, "children": 0}

    # Step 3 — embed only children
    child_texts = [text for text, _ in all_children]
    child_embeddings = await embed_batch(child_texts)

    # Step 4 — insert parents (no embedding), then children with parent_id
    async with get_connection() as conn:
        # Insert parents
        parent_db_ids = []
        for idx, parent_text in enumerate(parent_texts):
            parent_db_id = await conn.fetchval(
                """
                INSERT INTO chunks
                    (document_id, chunk_index, text, token_count,
                     chunk_strategy, page_number, embedding, parent_id)
                VALUES ($1, $2, $3, $4, 'parent', $5, NULL, NULL)
                RETURNING id
                """,
                document_id,
                idx,
                parent_text,
                count_tokens(parent_text),
                page_number,
            )
            parent_db_ids.append(parent_db_id)

        # Insert children with reference to their parent
        await conn.executemany(
            """
            INSERT INTO chunks
                (document_id, chunk_index, text, token_count,
                 chunk_strategy, page_number, embedding, parent_id)
            VALUES ($1, $2, $3, $4, 'child', $5, $6::vector, $7)
            """,
            [
                (
                    document_id,
                    i,
                    child_text,
                    count_tokens(child_text),
                    page_number,
                    str(embedding),
                    parent_db_ids[parent_idx],
                )
                for i, ((child_text, parent_idx), embedding)
                in enumerate(zip(all_children, child_embeddings))
            ],
        )

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Parent-child ingestion | doc_id={} | parents={} | children={} | {:.0f}ms",
        document_id,
        len(parent_texts),
        len(all_children),
        latency_ms,
    )

    return {
        "parents": len(parent_texts),
        "children": len(all_children),
        "latency_ms": round(latency_ms),
    }


async def retrieve_with_parent(
    question: str,
    top_k: int = 3,
    doc_type: str | None = "regulatory",
    min_similarity: float = 0.0,
) -> list[str]:
    """
    Retrieve using child chunks, then expand to parent context.

    Pipeline:
    1. Embed the question
    2. Search only among child chunks (embedding IS NOT NULL, strategy='child')
    3. For each found child, fetch its parent chunk (full context)
    4. Deduplicate parents (multiple children may share one parent)
    5. Return formatted parent texts for LLM injection

    This gives the best of both worlds:
    - Retrieval precision from small child chunks
    - Generation context from large parent chunks

    Args:
        question: User question.
        top_k: Number of parent chunks to return (not children).
        doc_type: Filter by document type.
        min_similarity: Minimum cosine similarity for child retrieval.

    Returns:
        Formatted parent chunk strings with source metadata.
    """
    from src.embeddings.embedder import embed_text

    start = time.perf_counter()

    # Step 1 — embed question
    query_embedding = await embed_text(question)
    embedding_str = str(query_embedding)

    # Step 2 — search among child chunks only
    if doc_type:
        query = """
            SELECT c.id, c.parent_id, c.text,
                   1 - (c.embedding <=> $1::vector) AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.chunk_strategy = 'child'
              AND c.embedding IS NOT NULL
              AND d.doc_type = $3
              AND 1 - (c.embedding <=> $1::vector) >= $4
            ORDER BY c.embedding <=> $1::vector
            LIMIT $2
        """
        params = (embedding_str, top_k * 3, doc_type, min_similarity)
    else:
        query = """
            SELECT c.id, c.parent_id, c.text,
                   1 - (c.embedding <=> $1::vector) AS similarity
            FROM chunks c
            WHERE c.chunk_strategy = 'child'
              AND c.embedding IS NOT NULL
              AND 1 - (c.embedding <=> $1::vector) >= $3
            ORDER BY c.embedding <=> $1::vector
            LIMIT $2
        """
        params = (embedding_str, top_k * 3, min_similarity)

    async with get_connection() as conn:
        child_rows = await conn.fetch(query, *params)

        if not child_rows:
            logger.warning("No child chunks found for question='{}'", question[:50])
            return []

        # Step 3 — fetch unique parents
        # Multiple children may point to the same parent — deduplicate
        seen_parent_ids = set()
        parent_ids_ordered = []
        for row in child_rows:
            pid = row["parent_id"]
            if pid and pid not in seen_parent_ids:
                seen_parent_ids.add(pid)
                parent_ids_ordered.append(pid)
            if len(parent_ids_ordered) >= top_k:
                break

        # Fetch parent texts in order
        parent_rows = await conn.fetch(
            """
            SELECT c.id, c.text, c.page_number, c.section_title, d.filename
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id = ANY($1::int[])
            """,
            parent_ids_ordered,
        )

    # Reorder parents to match the child retrieval order
    parent_map = {row["id"]: row for row in parent_rows}
    ordered_parents = [
        parent_map[pid] for pid in parent_ids_ordered if pid in parent_map
    ]

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Parent-child retrieval | q='{}' | children_found={} | parents={} | {:.0f}ms",
        question[:50],
        len(child_rows),
        len(ordered_parents),
        latency_ms,
    )

    # Format with source metadata
    formatted = []
    for row in ordered_parents:
        source = f"[{row['filename']}"
        if row["page_number"]:
            source += f", Page {row['page_number']}"
        if row["section_title"]:
            source += f", {row['section_title']}"
        source += "]"
        formatted.append(f"{source}\n{row['text']}")

    return formatted