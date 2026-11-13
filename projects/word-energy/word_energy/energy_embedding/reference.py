"""Small continuous references used before MCMC or learned joint training."""

from __future__ import annotations

from itertools import product
from dataclasses import dataclass
import numpy as np

from word_energy.numerics import logsumexp

from .system import ActiveWindowSystem


@dataclass(frozen=True)
class GaussianHermiteJointReference:
    """Exact-for-quadrature result for a small normalised Gaussian window."""

    log_z: float
    free_energy: float
    readout: np.ndarray
    component_assignments: int
    quadrature_points: int


def gaussian_hermite_joint_reference(
    system: ActiveWindowSystem,
    tokens: np.ndarray,
    *,
    order: int = 8,
    max_terms: int = 65_536,
) -> GaussianHermiteJointReference:
    """Evaluate normalised joint F and all R edges with Gauss--Hermite terms.

    This is deliberately a tiny D=1 reference, not a sampler.  It integrates
    the product of normalised unary mixtures and the bounded interaction:
    ``Z_rel = E_product_unary[exp(lambda * pair_kernel_sum)]``.  Therefore
    ``F=-log(Z_rel)`` must be zero when ``coupling=0``.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.ndim != 1 or len(tokens) < 1:
        raise ValueError("tokens must be a non-empty one-dimensional ID array")
    if system.landscape.dimension != 1:
        raise ValueError("The first Gaussian joint reference is restricted to D=1")
    if order < 2:
        raise ValueError("Gauss--Hermite order must be at least two")
    if np.any(tokens < 0) or np.any(tokens >= system.landscape.vocabulary_size):
        raise ValueError("Token ID is outside the landscape vocabulary")

    window, components = len(tokens), system.landscape.components
    assignment_count = components**window
    point_count = order**window
    if assignment_count * point_count > max_terms:
        raise ValueError("Quadrature term budget exceeded; refine in chunks or use a smaller toy")

    nodes, node_weights = np.polynomial.hermite.hermgauss(order)
    node_grid = np.asarray(list(product(nodes, repeat=window)), dtype=np.float64)
    quadrature_weights = np.prod(
        np.asarray(list(product(node_weights / np.sqrt(np.pi), repeat=window)), dtype=np.float64), axis=1
    )
    assignments = np.asarray(list(product(range(components), repeat=window)), dtype=np.int64)
    token_means = system.landscape.means[tokens]
    token_variances = system.landscape.variances[tokens]
    positions = np.broadcast_to(np.arange(window), assignments.shape)
    means = token_means[positions, assignments]
    standard_deviations = np.sqrt(token_variances[positions, assignments])
    states = means[:, None, :, :] + np.sqrt(2.0) * standard_deviations[:, None, :, :] * node_grid[None, :, :, None]

    mixture_weights = np.prod(
        np.asarray(
            [system.landscape.weights(int(token))[assignments[:, position]] for position, token in enumerate(tokens)]
        ),
        axis=0,
    )
    interaction = np.zeros((assignment_count, point_count), dtype=np.float64)
    kernels: dict[tuple[int, int], np.ndarray] = {}
    if window > 1:
        for left in range(window):
            for right in range(left + 1, window):
                kernel = system.kernel(states[:, :, left], states[:, :, right])
                kernels[(left, right)] = kernel
                interaction += kernel
        interaction *= system.coupling / (window - 1)
    log_terms = np.log(mixture_weights)[:, None] + np.log(quadrature_weights)[None, :] + interaction
    log_z = float(logsumexp(log_terms))
    normalised_terms = np.exp(log_terms - log_z)
    readout = np.eye(window, dtype=np.float64)
    for (left, right), kernel in kernels.items():
        value = float(np.sum(normalised_terms * kernel))
        readout[left, right] = readout[right, left] = value
    return GaussianHermiteJointReference(
        log_z=log_z,
        free_energy=-log_z,
        readout=readout,
        component_assignments=assignment_count,
        quadrature_points=point_count,
    )


def grid_reference_1d(system: ActiveWindowSystem, tokens: np.ndarray, grid: np.ndarray) -> tuple[float, np.ndarray]:
    """Grid estimate of log Z and R for D=1. Intended only for tiny reference windows."""
    if system.landscape.dimension != 1:
        raise ValueError("This reference is intentionally limited to D=1")
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("Grid needs at least two 1-D points")
    states = np.array(list(product(grid, repeat=len(tokens))), dtype=np.float64)[..., None]
    energies = np.array([system.joint_energy(tokens, state) for state in states])
    spacing = float(grid[1] - grid[0])
    log_z = float(logsumexp(-energies) + len(tokens) * np.log(abs(spacing)))
    weights = np.exp(-energies - logsumexp(-energies))
    samples = states
    readout = np.zeros((len(tokens), len(tokens)), dtype=np.float64)
    np.fill_diagonal(readout, 1.0)
    for left in range(len(tokens)):
        for right in range(left + 1, len(tokens)):
            value = float(np.sum(weights * system.kernel(samples[:, left], samples[:, right])))
            readout[left, right] = readout[right, left] = value
    return log_z, readout

