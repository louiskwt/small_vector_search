"""Tests for metrics.py.

    pytest tests/test_metrics.py -v

Tiny/huge values: always pass abs=0 so pytest's default abs=1e-12
cannot hide an underflow-to-zero.
"""

import math

import numpy as np
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


def test_cosine_is_symmetric():
    a, b = [0.2, 0.8, 0.1], [0.9, 0.1, 0.3]
    assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))


def test_euclidean_with_1d_vectors():
    assert euclidean_distance([3], [4]) == pytest.approx(1.0)


def test_cosine_with_1d_vectors():
    assert cosine_similarity([3], [4]) == pytest.approx(1.0)


def test_cosine_with_parallel_vectors():
    assert cosine_similarity([1, 1], [2, 2]) == pytest.approx(1.0)


def test_mixed_list_and_ndarray():
    assert euclidean_distance([3.0, 4.0], np.array([0.0, 0.0])) == pytest.approx(5.0)
    assert cosine_similarity([1.0, 0.0], np.array([1.0, 0.0])) == pytest.approx(1.0)


def test_euclidean_with_small_floats():
    assert euclidean_distance([1e-8], [2e-8]) == pytest.approx(1e-8, rel=1e-5, abs=0)


def test_cosine_with_small_floats():
    assert cosine_similarity([1e-8], [2e-8]) == pytest.approx(1.0, rel=1e-7, abs=0)


def test_euclidean_with_underflow_limit():
    # |2e-300 - 1e-300| = 1e-300. Naive (diff**2) underflows to 0.
    assert euclidean_distance([1e-300], [2e-300]) == pytest.approx(1e-300, rel=1e-6, abs=0)


def test_cosine_with_underflow_limit():
    assert cosine_similarity([1e-300], [2e-300]) == pytest.approx(1.0, rel=1e-5, abs=0)


def test_euclidean_with_orthogonal_tiny_vectors():
    assert euclidean_distance([1e-15, 0.0], [0.0, 1e-15]) == pytest.approx(
        math.sqrt(2) * 1e-15, rel=1e-5, abs=0
    )


def test_cosine_with_orthogonal_tiny_vectors():
    assert cosine_similarity([1e-15, 0.0], [0.0, 1e-15]) == pytest.approx(0.0, abs=1e-12)


def test_euclidean_rejects_empty_vector():
    with pytest.raises(ValueError):
        euclidean_distance([], [1.0])
    with pytest.raises(ValueError):
        euclidean_distance([], [])


def test_cosine_rejects_empty_vector():
    with pytest.raises(ValueError):
        cosine_similarity([], [1.0])
    with pytest.raises(ValueError):
        cosine_similarity([], [])


def test_cosine_both_zero_vectors():
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == pytest.approx(0.0)


def test_euclidean_zero_vectors():
    assert euclidean_distance([0.0, 0.0], [0.0, 0.0]) == pytest.approx(0.0)


def test_cosine_overflow_parallel():
    # Naive dot(1e300, 1e300) overflows to inf; scaled cosine must stay 1.0.
    huge = [1e300, 1e300]
    assert cosine_similarity(huge, huge) == pytest.approx(1.0, rel=1e-9, abs=0)


def test_cosine_overflow_opposite():
    assert cosine_similarity([1e300, 1e300], [-1e300, -1e300]) == pytest.approx(-1.0, rel=1e-9, abs=0)


def test_euclidean_overflow():
    # Naive (1e300)**2 -> inf. True L2 is 1e300.
    assert euclidean_distance([1e300], [0.0]) == pytest.approx(1e300, rel=1e-9, abs=0)


def test_euclidean_overflow_two_components():
    # Naive 1e200**2 + 1e200**2 -> inf. True L2 is 1e200 * sqrt(2).
    assert euclidean_distance([1e200, 1e200], [0.0, 0.0]) == pytest.approx(
        1e200 * math.sqrt(2), rel=1e-9, abs=0
    )


# Optional later: NaN / Inf policy (reject vs propagate). Not required for v1.
