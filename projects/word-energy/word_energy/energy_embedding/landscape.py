"""Normalised multi-well word landscapes used by the Gaussian baseline."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from word_energy.numerics import logsumexp, softmax


@dataclass(frozen=True)
class GaussianMixtureLandscape:
    """One normalised diagonal Gaussian mixture for every vocabulary item.

    `logits`, `means` and `variances` are learnable parameters in a later
    training stage.  This reference only accepts already-valid variances so
    every call has an unambiguous normalised density.
    """

    logits: np.ndarray  # [vocabulary, components]
    means: np.ndarray  # [vocabulary, components, dimension]
    variances: np.ndarray  # [vocabulary, components, dimension]

    def __post_init__(self) -> None:
        logits = np.asarray(self.logits, dtype=np.float64)
        means = np.asarray(self.means, dtype=np.float64)
        variances = np.asarray(self.variances, dtype=np.float64)
        if logits.ndim != 2 or means.ndim != 3 or variances.shape != means.shape:
            raise ValueError("Expected logits[V,M], means[V,M,D], variances[V,M,D]")
        if means.shape[:2] != logits.shape:
            raise ValueError("Mixture shapes disagree")
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(variances)):
            raise ValueError("Landscape parameters must be finite")
        if np.any(variances <= 0):
            raise ValueError("Variances must be strictly positive")
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "variances", variances)

    @property
    def vocabulary_size(self) -> int:
        return int(self.logits.shape[0])

    @property
    def components(self) -> int:
        return int(self.logits.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.means.shape[2])

    def weights(self, token: int) -> np.ndarray:
        return softmax(self.logits[token])

    def log_density(self, token: int, points: np.ndarray) -> np.ndarray:
        """Evaluate log p_token(z) for `points[..., dimension]`."""
        points = np.asarray(points, dtype=np.float64)
        if points.shape[-1] != self.dimension:
            raise ValueError("Point dimension does not match landscape")
        centered = points[..., None, :] - self.means[token]
        component_log = -0.5 * (
            np.sum(centered * centered / self.variances[token], axis=-1)
            + np.sum(np.log(2 * np.pi * self.variances[token]), axis=-1)
        )
        return logsumexp(component_log + np.log(self.weights(token)), axis=-1)

    def energy(self, token: int, points: np.ndarray) -> np.ndarray:
        """The normalised unary energy U_w(z) = -log p_w(z)."""
        return -self.log_density(token, points)

    def sample_unary(self, token: int, count: int, rng: np.random.Generator) -> np.ndarray:
        components = rng.choice(self.components, size=count, p=self.weights(token))
        noise = rng.normal(size=(count, self.dimension))
        return self.means[token, components] + noise * np.sqrt(self.variances[token, components])

