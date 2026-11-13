"""Continuous Gaussian-mixture energy-landscape baseline."""

from .landscape import GaussianMixtureLandscape
from .overlap import analytic_overlap
from .system import ActiveWindowSystem

__all__ = ["ActiveWindowSystem", "GaussianMixtureLandscape", "analytic_overlap"]

from .landscape import GaussianMixtureLandscape
from .overlap import analytic_overlap, compatibility_energy
from .reference import GaussianHermiteJointReference, gaussian_hermite_joint_reference
from .system import ActiveWindowSystem

__all__ = [
    "ActiveWindowSystem",
    "GaussianHermiteJointReference",
    "GaussianMixtureLandscape",
    "analytic_overlap",
    "compatibility_energy",
    "gaussian_hermite_joint_reference",
]
