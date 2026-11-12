"""Joint active-window energy and a shared-sample relevance readout."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .landscape import GaussianMixtureLandscape


def normalised_metric(raw: np.ndarray, rho: float = 0.05) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("Metric parameter must be square")
    if not 0 < rho < 1:
        raise ValueError("rho must lie between zero and one")
    gram = raw.T @ raw
    trace = float(np.trace(gram))
    if trace <= 1e-16:
        raise ValueError("Metric parameter has near-zero norm")
    dimension = raw.shape[0]
    return (1 - rho) * dimension * gram / trace + rho * np.eye(dimension)


@dataclass(frozen=True)
class ActiveWindowSystem:
    landscape: GaussianMixtureLandscape
    raw_metric: np.ndarray
    coupling: float = 1.0
    length_scale: float = 1.0
    rho: float = 0.05

    def __post_init__(self) -> None:
        if self.coupling < 0 or self.length_scale <= 0:
            raise ValueError("coupling must be non-negative and length scale positive")
        metric = normalised_metric(self.raw_metric, self.rho)
        if metric.shape != (self.landscape.dimension, self.landscape.dimension):
            raise ValueError("Metric and landscape dimension disagree")

    @property
    def metric(self) -> np.ndarray:
        return normalised_metric(self.raw_metric, self.rho)

    def kernel(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
        quadratic = np.einsum("...i,ij,...j->...", delta, self.metric, delta)
        return np.exp(-quadratic / (2 * self.length_scale**2))

    def joint_energy(self, tokens: np.ndarray, states: np.ndarray) -> float:
        """Energy of one window; states has shape [window, dimension]."""
        tokens = np.asarray(tokens, dtype=np.int64)
        states = np.asarray(states, dtype=np.float64)
        if states.shape != (len(tokens), self.landscape.dimension):
            raise ValueError("State shape does not match the active window")
        unary = sum(float(self.landscape.energy(int(token), state)) for token, state in zip(tokens, states))
        if len(tokens) <= 1:
            return unary
        pair_sum = sum(
            self.kernel(states[left], states[right])
            for left in range(len(tokens))
            for right in range(left + 1, len(tokens))
        )
        return float(unary - self.coupling * pair_sum / (len(tokens) - 1))

    def readout(self, joint_samples: np.ndarray) -> np.ndarray:
        """R_ij from one shared set of joint samples, never pairwise re-equilibration."""
        joint_samples = np.asarray(joint_samples, dtype=np.float64)
        if joint_samples.ndim != 3 or joint_samples.shape[-1] != self.landscape.dimension:
            raise ValueError("Expected samples[draw, window, dimension]")
        window = joint_samples.shape[1]
        output = np.eye(window, dtype=np.float64)
        for left in range(window):
            for right in range(left + 1, window):
                value = float(np.mean(self.kernel(joint_samples[:, left], joint_samples[:, right])))
                output[left, right] = output[right, left] = value
        return output

