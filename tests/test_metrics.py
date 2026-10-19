"""Starter tests for metrics.py.

Run from the repo root:
    pip install -e ".[dev]"
    pytest tests/test_metrics.py -v
"""

import pytest
import numpy as np
import math
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


def test_euclidean_distance_with_np_arrays():
    assert euclidean_distance(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(0.0)


def test_cosine_with_np_array():
    assert cosine_similarity(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)


def test_euclidean_is_symmetric():
    assert euclidean_distance([0.0, 0.0], [3.0, 4.0]) == euclidean_distance([3.0, 4.0], [0.0, 0.0])


# - single-dimension vectors [3] vs [4]
def test_euclidean_with_1D_vectors():
    assert euclidean_distance([3], [4]) == pytest.approx(1.0)


def test_cosine_with_1D_vectors():
    assert cosine_similarity([3], [4]) == pytest.approx(1.0)


# - cosine of parallel but different-length vectors: [1,1] vs [2,2] -> 1.0
def test_cosine_with_parallel_vectors():
    assert cosine_similarity([1, 1], [2, 2]) == pytest.approx(1.0)


# - very small values (numerical stability)
def test_euclidean_with_small_floats():
    assert euclidean_distance([1e-8], [2e-8]) == pytest.approx(1e-8, rel=1e-5) 


def test_cosine_with_small_floats():
    assert cosine_similarity([1e-8], [2e-8]) == pytest.approx(1.0, rel=1e-7) 


def test_euclidean_with_underflow_limit():
    assert euclidean_distance([1e-300], [2e-300]) == pytest.approx(2e-300, rel=1e-3)


def test_cosine_with_underflow_limit():
    assert cosine_similarity([1e-300], [2e-300]) == pytest.approx(1.0, rel=1e-5)


def test_euclidean_with_orthogonal_tiny_vectors():
    assert euclidean_distance([1e-15, 0.0], [0.0, 1e-15]) == pytest.approx(math.sqrt(2) * 1e-15, rel=1e-5)


def test_cosine_with_orthogonal_tiny_vectors():
    assert cosine_similarity([1e-15, 0.0], [0.0, 1e-15]) == pytest.approx(0.0, rel=1e-7)


