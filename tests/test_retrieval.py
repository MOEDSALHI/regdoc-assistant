# tests/test_retrieval.py
from src.embeddings.chunker import chunk_by_article, select_chunking_strategy
from src.rag.ingestion import extract_text_from_string

REGULATORY_TEXT = """
Article 5 - Conservation des donnees

Les donnees personnelles doivent etre conservees sous une forme permettant
l'identification des personnes concernees pendant une duree n'excedant pas
celle necessaire aux finalites pour lesquelles elles sont traitees.

Article 6 - Base legale

Le traitement est licite seulement si la personne concernee a donne son
consentement au traitement de ses donnees pour une ou plusieurs finalites.
"""


def test_extract_text_from_string_returns_pages():
    pages = extract_text_from_string("Texte de test suffisamment long pour etre utile.")
    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert "Texte de test" in pages[0]["text"]


def test_extract_text_from_string_page_number():
    pages = extract_text_from_string("contenu quelconque pour le test")
    assert pages[0]["page_number"] == 1


def test_chunking_pipeline_produces_chunks():
    pages = extract_text_from_string(REGULATORY_TEXT)
    all_chunks = []
    for page in pages:
        chunks = select_chunking_strategy(
            text=page["text"],
            document_type="regulatory",
            metadata={"filename": "test.pdf", "page_number": page["page_number"]},
        )
        all_chunks.extend(chunks)
    assert len(all_chunks) == 2  # two articles
    assert all_chunks[0].text.startswith("Article 5")
    assert all_chunks[1].text.startswith("Article 6")


def test_chunking_pipeline_metadata_propagation():
    pages = extract_text_from_string(REGULATORY_TEXT)
    chunks = select_chunking_strategy(
        text=pages[0]["text"],
        document_type="regulatory",
        metadata={"filename": "rgpd.pdf", "page_number": 1},
    )
    assert all(c.metadata["filename"] == "rgpd.pdf" for c in chunks)
    assert all(c.metadata["page_number"] == 1 for c in chunks)


def test_chunking_preserves_article_content():
    chunks = chunk_by_article(REGULATORY_TEXT)
    combined = " ".join(c.text for c in chunks)
    assert "Conservation des donnees" in combined
    assert "Base legale" in combined
