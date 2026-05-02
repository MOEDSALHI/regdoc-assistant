# tests/test_hybrid_search.py
import pytest
from src.rag.hybrid_search import _bm25_search, _reciprocal_rank_fusion, RankedChunk
from src.db.chunks_repo import ChunkRecord


def _make_chunk(id: int, text: str) -> ChunkRecord:
    return ChunkRecord(
        id=id, document_id=1, chunk_index=id,
        text=text, token_count=len(text.split()),
        chunk_strategy="by_article", page_number=1,
        section_title=None, similarity=0.0,
        filename=f"doc{id}.txt",
    )


CORPUS = [
    _make_chunk(1, "Article 5 - Les donnees personnelles doivent etre traitees de maniere licite"),
    _make_chunk(2, "Article 17 - Droit a l effacement des donnees personnelles"),
    _make_chunk(3, "Article 35 - DPIA obligatoire pour les traitements a risque eleve RSSI"),
    _make_chunk(4, "CNIL recommandation - Conservation des logs pendant 6 mois minimum"),
    _make_chunk(5, "ANSSI guide - Authentification forte recommandee pour les systemes sensibles"),
]


# ── BM25 ──────────────────────────────────────────────────────────────────────

def test_bm25_returns_results():
    results = _bm25_search("DPIA RSSI", CORPUS, top_k=3)
    assert len(results) > 0


def test_bm25_exact_term_ranks_first():
    results = _bm25_search("DPIA RSSI", CORPUS, top_k=5)
    top_chunk, top_score = results[0]
    assert "DPIA" in top_chunk.text
    assert top_score > 0


def test_bm25_filters_zero_scores():
    # "xyz" not in any chunk → all scores = 0 → filtered out
    results = _bm25_search("xyz123notexist", CORPUS, top_k=5)
    assert len(results) == 0


def test_bm25_respects_top_k():
    results = _bm25_search("donnees", CORPUS, top_k=2)
    assert len(results) <= 2


def test_bm25_scores_positive():
    results = _bm25_search("logs conservation", CORPUS, top_k=5)
    assert all(score > 0 for _, score in results)


# ── RRF ───────────────────────────────────────────────────────────────────────

def _make_vector_results() -> list[ChunkRecord]:
    """Simulate vector search results ordered by cosine similarity."""
    results = list(CORPUS)
    results[0] = _make_chunk(3, CORPUS[2].text)
    results[0].similarity = 0.92
    return [results[0], CORPUS[0], CORPUS[3]]


def test_rrf_returns_ranked_chunks():
    vector = [CORPUS[2], CORPUS[0], CORPUS[3]]  # DPIA first
    bm25 = [(CORPUS[2], 5.2), (CORPUS[3], 3.1)]  # DPIA also first in BM25
    result = _reciprocal_rank_fusion(vector, bm25, top_k=3)
    assert len(result) <= 3
    assert all(isinstance(r, RankedChunk) for r in result)


def test_rrf_convergent_signals_rank_highest():
    # chunk appears #1 in both vector and BM25 → should be #1 in RRF
    vector = [CORPUS[2], CORPUS[0], CORPUS[3]]
    bm25   = [(CORPUS[2], 8.0), (CORPUS[0], 2.0)]
    result = _reciprocal_rank_fusion(vector, bm25, top_k=3)
    assert result[0].text == CORPUS[2].text


def test_rrf_score_formula():
    # Verify RRF score: rank1 in both lists = 1/(60+1) + 1/(60+1) = 0.03279
    vector = [CORPUS[0]]
    bm25   = [(CORPUS[0], 1.0)]
    result = _reciprocal_rank_fusion(vector, bm25, top_k=1, k=60)
    expected = 1/(60+1) + 1/(60+1)
    assert abs(result[0].rrf_score - expected) < 1e-6


def test_rrf_chunk_in_one_list_only():
    # chunk only in vector (not BM25) → lower RRF score
    vector = [CORPUS[0], CORPUS[1]]
    bm25   = [(CORPUS[0], 5.0)]  # only chunk 0
    result = _reciprocal_rank_fusion(vector, bm25, top_k=2, k=60)
    # chunk[0] appears in both → higher score than chunk[1] (vector only)
    assert result[0].text == CORPUS[0].text


def test_rrf_respects_top_k():
    vector = list(CORPUS)
    bm25   = [(c, 1.0) for c in CORPUS]
    result = _reciprocal_rank_fusion(vector, bm25, top_k=2)
    assert len(result) == 2