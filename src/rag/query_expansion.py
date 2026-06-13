# src/rag/query_expansion.py
import time

from loguru import logger

from src.services.llm_client import chat_complete


async def expand_query(question: str, n_variants: int = 3) -> list[str]:
    """
    Generate semantic variants of a question using the LLM.

    Why: different phrasings of the same question retrieve different chunks.
    Merging results from multiple phrasings improves recall.

    The original question is always included in the returned list
    to ensure we don't lose the original intent.

    Args:
        question: Original user question.
        n_variants: Number of additional variants to generate.

    Returns:
        List of [original_question] + [n_variants], deduplicated.
    """
    prompt = f"""Generate {n_variants} different phrasings of this question about GDPR/RGPD regulations.
Each phrasing should use different vocabulary but keep the same meaning.
Return ONLY the phrasings, one per line, no numbering, no explanation.

Question: {question}"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a GDPR regulatory expert. Generate concise question variants "
                "that would help retrieve relevant regulatory passages. "
                "Use French regulatory terminology."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    raw = await chat_complete(messages, temperature=0.7, max_tokens=200)

    # Parse variants — one per line, strip empty lines
    variants = [
        line.strip()
        for line in raw.strip().split("\n")
        if line.strip() and line.strip() != question
    ][:n_variants]

    all_queries = [question] + variants

    logger.info(
        "Query expansion | original='{}' | variants={}",
        question[:50],
        len(variants),
    )
    for i, q in enumerate(all_queries):
        logger.debug("  Query {}: {}", i, q[:70])

    return all_queries


async def retrieve_with_expansion(
    question: str,
    top_k: int = 5,
    n_variants: int = 3,
    doc_type: str | None = "regulatory",
) -> list[str]:
    """
    Retrieve chunks using multiple query variants and merge with RRF.

    Pipeline:
    1. Generate n_variants phrasings of the question
    2. Embed each phrasing separately
    3. Run pgvector search for each embedding
    4. Merge all result lists with RRF

    This is also called "RAG-Fusion" when n_variants >= 3.

    Args:
        question: Original user question.
        top_k: Final number of chunks to return.
        n_variants: Number of additional query variants.
        doc_type: Filter by document type.

    Returns:
        Formatted chunk strings merged and ranked by RRF.
    """
    from src.db.chunks_repo import ChunkRecord, similarity_search
    from src.embeddings.embedder import embed_text

    start = time.perf_counter()

    # Step 1 — generate query variants
    queries = await expand_query(question, n_variants=n_variants)

    # Step 2+3 — embed and search each variant
    all_result_lists: list[list[ChunkRecord]] = []
    for query in queries:
        embedding = await embed_text(query)
        results = await similarity_search(
            query_embedding=embedding,
            top_k=top_k * 2,
            doc_type=doc_type,
            min_similarity=0.0,
        )
        all_result_lists.append(results)

    # Step 4 — RRF fusion across all result lists

    # Convert to format expected by _reciprocal_rank_fusion
    # We merge pairwise: list1 + list2, then result + list3, etc.
    if not all_result_lists:
        return []

    # Collect all unique chunks with their best rank across queries
    chunk_ranks: dict[int, list[int]] = {}  # chunk_id -> [rank in each list]
    chunk_map: dict[int, ChunkRecord] = {}

    for result_list in all_result_lists:
        for rank, chunk in enumerate(result_list):
            chunk_map[chunk.id] = chunk
            if chunk.id not in chunk_ranks:
                chunk_ranks[chunk.id] = []
            chunk_ranks[chunk.id].append(rank + 1)

    # RRF score: sum of 1/(k + rank) across all query lists
    k = 60
    rrf_scores = {
        chunk_id: sum(1.0 / (k + r) for r in ranks) for chunk_id, ranks in chunk_ranks.items()
    }

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Query expansion retrieval | queries={} | unique_chunks={} | final={} | {:.0f}ms",
        len(queries),
        len(chunk_map),
        min(top_k, len(sorted_ids)),
        latency_ms,
    )

    # Format top_k results
    results = []
    for chunk_id in sorted_ids[:top_k]:
        chunk = chunk_map[chunk_id]
        source = f"[{chunk.filename}"
        if chunk.page_number:
            source += f", Page {chunk.page_number}"
        source += "]"
        results.append(f"{source}\n{chunk.text}")

    return results


async def hyde_retrieve(
    question: str,
    top_k: int = 5,
    doc_type: str | None = "regulatory",
) -> list[str]:
    """
    HyDE: Hypothetical Document Embedding retrieval.

    Instead of embedding the question directly, we ask the LLM to generate
    a hypothetical answer document, then embed THAT document.

    Why it works: the hypothetical document uses the same vocabulary as
    real regulatory chunks, making it much closer in the embedding space.

    Example:
        Question: "Quels droits a une personne concernée ?"
        Hypothetical doc: "Les personnes concernées disposent du droit d'accès
                          (Art. 15), rectification (Art. 16), effacement (Art. 17)..."
        → embedding(hypothetical_doc) >> embedding(question) for retrieval

    Risk: if the LLM hallucinates facts in the hypothetical doc, we may retrieve
    irrelevant chunks. Mitigated by using low temperature (0.3).

    Args:
        question: User question.
        top_k: Number of chunks to return.
        doc_type: Filter by document type.

    Returns:
        Formatted chunk strings retrieved using the hypothetical document embedding.
    """
    from src.db.chunks_repo import similarity_search
    from src.embeddings.embedder import embed_text

    start = time.perf_counter()

    # Step 1 — generate hypothetical document
    hyde_prompt = f"""Write a short passage (3-5 sentences) from a French regulatory document
(RGPD, CNIL recommendation, or ANSSI guide) that would directly answer this question.
Write as if you are quoting from an official regulatory text.
Do NOT say "this passage answers the question" — just write the passage directly.

Question: {question}"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a GDPR regulatory document. "
                "Generate a plausible regulatory passage that answers the question. "
                "Use official regulatory French vocabulary and article references."
            ),
        },
        {"role": "user", "content": hyde_prompt},
    ]

    hypothetical_doc = await chat_complete(
        messages,
        temperature=0.3,  # low temp — we want plausible, not creative
        max_tokens=200,
    )

    logger.info(
        "HyDE | question='{}' | hypothetical_doc='{}'",
        question[:50],
        hypothetical_doc[:80],
    )

    # Step 2 — embed the hypothetical document (not the question)
    embedding = await embed_text(hypothetical_doc)

    # Step 3 — retrieve using the hypothetical document embedding
    records = await similarity_search(
        query_embedding=embedding,
        top_k=top_k,
        doc_type=doc_type,
        min_similarity=0.0,
    )

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "HyDE retrieval | results={} | best={:.4f} | {:.0f}ms",
        len(records),
        records[0].similarity if records else 0,
        latency_ms,
    )

    # Format results
    formatted = []
    for r in records:
        source = f"[{r.filename}"
        if r.page_number:
            source += f", Page {r.page_number}"
        source += "]"
        formatted.append(f"{source}\n{r.text}")

    return formatted
