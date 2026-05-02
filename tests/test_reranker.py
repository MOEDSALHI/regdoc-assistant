# tests/test_reranker.py
import pytest
from src.rag.reranker import rerank


def test_rerank_returns_sorted_by_score():
    question = "What is DPIA?"
    chunks = [
        "DPIA is required for high-risk data processing operations.",
        "The weather is sunny today.",
        "Article 35 requires a DPIA for high risk treatments.",
    ]
    results = rerank(question, chunks)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_relevant_chunk_scores_higher():
    question = "DPIA obligatoire traitement risque eleve"
    relevant = "Le DPIA est obligatoire pour les traitements a risque eleve selon l article 35."
    irrelevant = "L authentification forte est recommandee pour les systemes sensibles."
    results = rerank(question, [irrelevant, relevant])
    # The relevant chunk should rank first regardless of input order
    assert relevant in results[0][0]


def test_rerank_empty_input():
    results = rerank("any question", [])
    assert results == []


def test_rerank_single_chunk():
    results = rerank("DPIA", ["Article 35 DPIA obligatoire."])
    assert len(results) == 1
    assert isinstance(results[0][1], float)


def test_rerank_respects_top_k():
    chunks = [f"chunk number {i}" for i in range(6)]
    results = rerank("chunk", chunks, top_k=3)
    assert len(results) == 3


def test_rerank_scores_are_floats():
    results = rerank("test", ["chunk A", "chunk B"])
    assert all(isinstance(score, float) for _, score in results)


def test_rerank_irrelevant_chunk_negative_score():
    question = "DPIA obligatoire article 35 RGPD"
    irrelevant = "La recette de la tarte tatin necessite des pommes et du beurre."
    results = rerank(question, [irrelevant])
    # Irrelevant chunk should have a negative logit score
    assert results[0][1] < 0