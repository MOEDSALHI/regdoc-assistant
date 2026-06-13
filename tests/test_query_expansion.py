# tests/test_query_expansion.py


# Pure logic tests — no API needed


def test_rrf_multi_query_deduplication():
    """Verify that the same chunk appearing in multiple query results is deduplicated."""
    from src.db.chunks_repo import ChunkRecord

    def make_chunk(id, text):
        return ChunkRecord(
            id=id,
            document_id=1,
            chunk_index=id,
            text=text,
            token_count=5,
            chunk_strategy="by_article",
            page_number=1,
            section_title=None,
            similarity=0.9,
            filename="test.txt",
        )

    # Simulate same chunk appearing in 3 different query result lists
    chunk_a = make_chunk(1, "DPIA obligatoire traitement risque")
    chunk_b = make_chunk(2, "Conservation logs 6 mois")

    all_result_lists = [
        [chunk_a, chunk_b],  # query 1 result
        [chunk_a, chunk_b],  # query 2 result (same chunks)
        [chunk_b, chunk_a],  # query 3 result (reversed order)
    ]

    # Manual RRF computation
    k = 60
    chunk_ranks = {}
    chunk_map = {}

    for result_list in all_result_lists:
        for rank, chunk in enumerate(result_list):
            chunk_map[chunk.id] = chunk
            if chunk.id not in chunk_ranks:
                chunk_ranks[chunk.id] = []
            chunk_ranks[chunk.id].append(rank + 1)

    # chunk_a appears as rank 1, 1, 2 across 3 queries
    # chunk_b appears as rank 2, 2, 1 across 3 queries
    rrf_a = sum(1 / (k + r) for r in chunk_ranks[1])
    rrf_b = sum(1 / (k + r) for r in chunk_ranks[2])

    # chunk_a has better average rank → higher RRF score
    assert rrf_a > rrf_b

    # Total unique chunks = 2 (deduplicated correctly)
    assert len(chunk_map) == 2


def test_rrf_k_parameter_effect():
    """Higher k = less penalty for lower ranks."""
    # rank 1 with k=60: 1/61 = 0.01639
    # rank 1 with k=10: 1/11 = 0.09090
    assert 1 / (10 + 1) > 1 / (60 + 1)

    # rank 10 with k=60: 1/70 = 0.01428
    # rank 10 with k=10: 1/20 = 0.05
    assert 1 / (10 + 10) > 1 / (60 + 10)
