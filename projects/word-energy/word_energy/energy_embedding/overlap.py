"""Analytic landscape overlap for diagonal Gaussian mixtures."""

from __future__ import annotations

import numpy as np

from .landscape import GaussianMixtureLandscape


def _normal_at_mean_difference(mean_a: np.ndarray, mean_b: np.ndarray, covariance: np.ndarray) -> float:
    dimension = mean_a.shape[0]
    delta = mean_a - mean_b
    log_value = -0.5 * (
        dimension * np.log(2 * np.pi)
        + np.sum(np.log(covariance))
        + np.sum(delta * delta / covariance)
    )
    return float(np.exp(log_value))


def analytic_overlap(landscape: GaussianMixtureLandscape, left: int, right: int) -> float:
    """Return ∫ p_left(z) p_right(z) dz exactly for diagonal mixtures."""
    result = 0.0
    for component_left, weight_left in enumerate(landscape.weights(left)):
        for component_right, weight_right in enumerate(landscape.weights(right)):
            result += float(weight_left * weight_right) * _normal_at_mean_difference(
                landscape.means[left, component_left],
                landscape.means[right, component_right],
                landscape.variances[left, component_left] + landscape.variances[right, component_right],
            )
    return result


def compatibility_energy(landscape: GaussianMixtureLandscape, left: int, right: int) -> float:
    overlap = analytic_overlap(landscape, left, right)
    return float(-np.log(overlap))

