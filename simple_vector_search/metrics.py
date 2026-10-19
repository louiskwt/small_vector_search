"""Distance / similarity functions used by VectorStore.query()."""

from typing import Union

import numpy as np

ArrayLike = Union[list, np.ndarray]


def _as_1d(v: ArrayLike) -> np.ndarray:
    """Turn a list or array into a 1-D float vector."""
    result = np.asarray(v, dtype=float).ravel()
    if result.size == 0: raise ValueError("The 1-D float vector cannot be empty.")
    return result


def euclidean_distance(a: ArrayLike, b: ArrayLike) -> float:
    """L2 distance: sqrt(sum((a_i - b_i)^2)). Smaller = closer."""
    a_i, b_i = _as_1d(a), _as_1d(b)
    if a_i.shape != b_i.shape: raise ValueError("Vectors must have the same shape.")
    distance = np.sqrt(np.sum((a_i - b_i) ** 2))
    return distance


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Cosine similarity in [-1, 1]. Larger = closer."""
    a_i, b_i = _as_1d(a), _as_1d(b)
    
    if a_i.shape != b_i.shape: raise ValueError("Vectors must have the same shape.")

    # handling extreme number
    max_a, max_b = np.max(np.abs(a_i)), np.max(np.abs(b_i))

    if max_a == 0.0 or max_b == 0.0:
        return 0.0

    scaled_a = a_i / max_a
    scaled_b = b_i / max_b    

    denom = np.sqrt(np.sum(scaled_a ** 2)) * np.sqrt(np.sum(scaled_b ** 2))
    return 0.0 if denom == 0 else float(np.dot(scaled_a, scaled_b) / denom)