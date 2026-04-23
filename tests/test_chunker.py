# tests/test_chunker.py
import pytest
from src.embeddings.chunker import (
    chunk_fixed_size,
    chunk_recursive,
    chunk_by_article,
    select_chunking_strategy,
    Chunk,
)

SAMPLE_RGPD = """
Article 5 - Principes relatifs au traitement

Les donnees doivent etre traitees de maniere licite, loyale et transparente.
Elles doivent etre collectees pour des finalites determinees et legitimes.

Article 6 - Liceiete du traitement

Le traitement n'est licite que si la personne a consenti ou si le traitement
est necessaire a l'execution d'un contrat.

Article 17 - Droit a l'effacement

La personne a le droit d'obtenir l'effacement de ses donnees personnelles.
"""

SHORT_TEXT = "Texte trop court."
PLAIN_TEXT = "Un texte sans marqueur d'article. Il contient plusieurs phrases. Chaque phrase apporte de l'information sur la protection des donnees personnelles."


def test_chunk_fixed_size_returns_chunks():
    chunks = chunk_fixed_size(SAMPLE_RGPD, chunk_size=100, overlap=10)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_fixed_size_respects_token_limit():
    chunks = chunk_fixed_size(SAMPLE_RGPD, chunk_size=50, overlap=5)
    assert all(c.token_count <= 55 for c in chunks)  # slight tolerance


def test_chunk_fixed_size_empty_input():
    chunks = chunk_fixed_size("", chunk_size=100, overlap=10)
    assert chunks == []


def test_chunk_recursive_returns_chunks():
    chunks = chunk_recursive(SAMPLE_RGPD, chunk_size=100, overlap=10)
    assert len(chunks) > 0


def test_chunk_recursive_metadata():
    meta = {"filename": "test.pdf", "doc_type": "regulatory"}
    chunks = chunk_recursive(SAMPLE_RGPD, metadata=meta)
    assert all(c.metadata["filename"] == "test.pdf" for c in chunks)
    assert all(c.metadata["chunk_strategy"] == "recursive" for c in chunks)


def test_chunk_by_article_detects_articles():
    chunks = chunk_by_article(SAMPLE_RGPD)
    assert len(chunks) == 3
    assert chunks[0].text.startswith("Article 5")
    assert chunks[1].text.startswith("Article 6")
    assert chunks[2].text.startswith("Article 17")


def test_chunk_by_article_strategy_tag():
    chunks = chunk_by_article(SAMPLE_RGPD)
    assert all(c.metadata["chunk_strategy"] == "by_article" for c in chunks)


def test_chunk_by_article_no_markers_fallback():
    # No article markers -> falls back to recursive
    chunks = chunk_by_article(PLAIN_TEXT)
    assert len(chunks) > 0
    # Should not crash


def test_chunk_index_sequential():
    chunks = chunk_by_article(SAMPLE_RGPD)
    indices = [c.index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_select_strategy_regulatory_uses_article():
    chunks = select_chunking_strategy(SAMPLE_RGPD, document_type="regulatory")
    assert len(chunks) == 3
    assert chunks[0].metadata["chunk_strategy"] == "by_article"


def test_select_strategy_generic_uses_fixed():
    chunks = select_chunking_strategy(PLAIN_TEXT, document_type="generic")
    assert len(chunks) > 0
    assert chunks[0].metadata["chunk_strategy"] == "fixed_size"


def test_chunk_token_count_positive():
    chunks = chunk_by_article(SAMPLE_RGPD)
    assert all(c.token_count > 0 for c in chunks)