"""Starter tests for metrics.py.

Run from the repo root:
    pip install -e ".[dev]"
    pytest tests/test_metrics.py -v
"""

import pytest

from simple_vector_search.metrics import cosine_similarity, euclidean_distance


def test_euclidean_identical_vectors_is_zero():
    assert euclidean_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_euclidean_known_distance():
    # 3-4-5 triangle: (0,0) vs (3,4) -> 5
    assert euclidean_distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(5.0)


def test_euclidean_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        euclidean_distance([1.0, 2.0], [1.0, 2.0, 3.0])


def test_cosine_identical_vectors_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors_is_minus_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector_does_not_crash():
    # Pick a defined behavior (recommended: 0.0) and stick to it.
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


def test_cosine_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])


# Further tests to add once the basics pass:
# - numpy arrays as inputs, not only lists
# - single-dimension vectors [3] vs [4]
# - very small values (numerical stability)
# - cosine of parallel but different-length vectors: [1,1] vs [2,2] -> 1.0
# - confirm euclidean is symmetric: d(a,b) == d(b,a)
