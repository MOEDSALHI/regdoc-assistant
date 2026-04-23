# tests/test_embedder.py
import pytest
from src.embeddings.embedder import cosine_similarity

# cosine_similarity is a pure function — no API call needed
# embed_text and embed_batch require API — tested in integration only


def test_cosine_similarity_identical_vectors():
    vec = [1.0, 0.0, 0.0, 0.5]
    assert cosine_similarity(vec, vec) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_orthogonal_vectors():
    vec_a = [1.0, 0.0]
    vec_b = [0.0, 1.0]
    assert cosine_similarity(vec_a, vec_b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_opposite_vectors():
    vec_a = [1.0, 0.0]
    vec_b = [-1.0, 0.0]
    assert cosine_similarity(vec_a, vec_b) == pytest.approx(-1.0, abs=1e-6)


def test_cosine_similarity_zero_vector():
    vec_a = [0.0, 0.0]
    vec_b = [1.0, 0.5]
    assert cosine_similarity(vec_a, vec_b) == 0.0


def test_cosine_similarity_known_values():
    # [1, 1] and [1, 0] -> cos(45°) = sqrt(2)/2 ≈ 0.7071
    vec_a = [1.0, 1.0]
    vec_b = [1.0, 0.0]
    assert cosine_similarity(vec_a, vec_b) == pytest.approx(0.7071, abs=1e-3)


def test_cosine_similarity_symmetry():
    vec_a = [0.3, 0.7, 0.1]
    vec_b = [0.8, 0.2, 0.5]
    assert cosine_similarity(vec_a, vec_b) == pytest.approx(
        cosine_similarity(vec_b, vec_a), abs=1e-9
    )