"""Train-token-only spectral initialization for the Ising research ablation."""
from __future__ import annotations

import hashlib
import numpy as np


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str(array.dtype).encode() + repr(array.shape).encode() + array.tobytes()).hexdigest()


def spectral_biases(train_tokens: np.ndarray, vocabulary_size: int) -> dict:
    """Construct deterministic bias coordinates; accepts no labels or other splits."""
    tokens = np.asarray(train_tokens)
    if tokens.ndim != 2 or tokens.shape[1] < 2 or not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError('train_tokens must be a 2D integer array with at least two positions')
    if vocabulary_size < 4 or tokens.size == 0 or tokens.min() < 0 or tokens.max() >= vocabulary_size:
        raise ValueError('invalid vocabulary or token IDs')
    counts = np.zeros((vocabulary_size, vocabulary_size), dtype=np.int64)
    for window in tokens:
        for i in range(len(window)):
            for j in range(len(window)):
                if i != j:
                    counts[window[i], window[j]] += 1
    matrix = np.log1p(counts.astype(np.float64))
    center = np.eye(vocabulary_size) - np.ones((vocabulary_size, vocabulary_size))/vocabulary_size
    centered = center @ matrix @ center
    values, vectors = np.linalg.eigh(centered)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:,order]
    tolerance = 1e-8 * max(1.0, float(np.linalg.norm(centered, ord=2)))
    if values[1] <= 0 or min(values[0]-values[1], values[1]-values[2]) <= tolerance:
        raise ValueError('top two positive eigenvalues not separated; no label-aware fallback')
    vectors = vectors[:, :2].copy()
    for axis in range(2):
        index = int(np.argmax(np.abs(vectors[:, axis])))
        if vectors[index, axis] < 0:
            vectors[:, axis] *= -1
    deviations = vectors - vectors.mean(axis=0)
    standard_deviations = deviations.std(axis=0, ddof=0)
    if np.any(standard_deviations <= 1e-12):
        raise ValueError('degenerate coordinate standard deviation')
    biases = 0.1 * deviations / standard_deviations
    permutation = np.random.default_rng(6061).permutation(vocabulary_size)
    return {'biases': biases, 'permuted_biases': biases[permutation], 'permutation': permutation, 'counts': counts, 'eigenvectors': vectors, 'eigenvalues': values, 'eigengap_tolerance': tolerance}
