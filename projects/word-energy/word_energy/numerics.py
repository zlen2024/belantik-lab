"""Small, dependency-light numerical helpers used by both research paths."""

from __future__ import annotations

import numpy as np


def logsumexp(values: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    """Stable NumPy log-sum-exp without depending on SciPy."""
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    summed = np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    result = maximum + np.log(summed)
    if axis is None:
        return np.asarray(result.squeeze(), dtype=np.float64)
    return np.squeeze(result, axis=axis)


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=axis, keepdims=True)

