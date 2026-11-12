"""Continuous Gaussian-mixture energy-landscape baseline."""

from .landscape import GaussianMixtureLandscape
from .overlap import analytic_overlap
from .system import ActiveWindowSystem

__all__ = ["ActiveWindowSystem", "GaussianMixtureLandscape", "analytic_overlap"]

