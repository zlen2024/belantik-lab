"""Small continuous references used before MCMC or learned joint training."""

from __future__ import annotations

from itertools import product
from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

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


@dataclass(frozen=True)
class GaussianHermiteCenterMarginal:
    """Weighted central-token states from one shared joint quadrature."""

    states: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class AdaptiveCenterRegionMass:
    """One-dimensional central region mass with adaptive boundary integration.

    The two context coordinates are still integrated with Gauss--Hermite
    nodes.  The central coordinate is integrated separately on either side of
    every train-derived sense-map boundary, so a discontinuous region
    indicator is not evaluated only at a fixed set of central nodes.
    """

    finance_mass: float
    total_mass: float
    finance_probability: float
    integration_error: float
    boundaries: tuple[float, ...]
    lower: float
    upper: float


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


def gaussian_hermite_center_marginal_1d(
    system: ActiveWindowSystem,
    tokens: np.ndarray,
    *,
    order: int = 12,
    max_terms: int = 65_536,
) -> GaussianHermiteCenterMarginal:
    """Return the exact-for-quadrature central marginal for a three-token window.

    The caller can apply a train-derived region map to these weighted states;
    no generator family label is used while constructing the marginal.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.shape != (3,):
        raise ValueError("The first contextual marginal reference expects exactly three tokens")
    if system.landscape.dimension != 1:
        raise ValueError("The first contextual marginal reference is restricted to D=1")
    assignments = np.asarray(list(product(range(system.landscape.components), repeat=3)), dtype=np.int64)
    nodes, node_weights = np.polynomial.hermite.hermgauss(order)
    node_grid = np.asarray(list(product(nodes, repeat=3)), dtype=np.float64)
    if len(assignments) * len(node_grid) > max_terms:
        raise ValueError("Quadrature term budget exceeded")
    quadrature_weights = np.prod(
        np.asarray(list(product(node_weights / np.sqrt(np.pi), repeat=3)), dtype=np.float64), axis=1
    )
    positions = np.broadcast_to(np.arange(3), assignments.shape)
    means = system.landscape.means[tokens][positions, assignments]
    standard_deviations = np.sqrt(system.landscape.variances[tokens][positions, assignments])
    states = means[:, None, :, :] + np.sqrt(2.0) * standard_deviations[:, None, :, :] * node_grid[None, :, :, None]
    mixture_weights = np.prod(
        np.asarray(
            [system.landscape.weights(int(token))[assignments[:, position]] for position, token in enumerate(tokens)]
        ),
        axis=0,
    )
    interaction = np.zeros((len(assignments), len(node_grid)), dtype=np.float64)
    for left in range(3):
        for right in range(left + 1, 3):
            interaction += system.kernel(states[:, :, left], states[:, :, right])
    log_terms = (
        np.log(mixture_weights)[:, None]
        + np.log(quadrature_weights)[None, :]
        + system.coupling * interaction / 2.0
    )
    normalised = np.exp(log_terms - logsumexp(log_terms))
    return GaussianHermiteCenterMarginal(states=states[:, :, 0, 0].reshape(-1), weights=normalised.reshape(-1))


def _sense_boundaries_1d(q_finance, q_river, lower: float, upper: float, grid_size: int) -> tuple[float, ...]:
    """Locate genuine crossings of two train-only density estimates.

    Region membership is defined by the strict relation ``q_finance >
    q_river``.  In particular, do not turn a very small absolute density
    difference in a Gaussian tail into a fake boundary: that made the old
    domain-refinement audit depend on its interval midpoint.
    """
    if grid_size < 3:
        raise ValueError("grid_size must be at least three")
    grid = np.linspace(lower, upper, grid_size)

    def difference(value: float) -> float:
        point = np.asarray([value], dtype=np.float64)
        return float(np.asarray(q_finance(point)).reshape(-1)[0] - np.asarray(q_river(point)).reshape(-1)[0])

    values = np.asarray([difference(value) for value in grid])
    signs = np.sign(values).astype(np.int8)
    roots: list[float] = []
    previous: int | None = None
    for index, sign in enumerate(signs):
        if sign == 0:
            continue
        if previous is not None and sign != signs[previous]:
            left, right = float(grid[previous]), float(grid[index])
            roots.append(float(brentq(difference, left, right, xtol=1e-13, rtol=1e-13)))
        previous = index
    deduplicated: list[float] = []
    for root in sorted(roots):
        if not deduplicated or abs(root - deduplicated[-1]) > 1e-9:
            deduplicated.append(root)
    return tuple(deduplicated)


def adaptive_center_region_mass_1d(
    system: ActiveWindowSystem,
    tokens: np.ndarray,
    q_finance,
    q_river,
    *,
    context_order: int = 12,
    lower: float = -12.0,
    upper: float = 12.0,
    boundary_grid_size: int = 1025,
    boundaries: tuple[float, ...] | None = None,
    epsabs: float = 1e-9,
    epsrel: float = 1e-9,
) -> AdaptiveCenterRegionMass:
    """Integrate the central finance-region probability for a three-token window.

    This is a validation reference, not a sampler: it marginalises the two
    context positions with fixed Gauss--Hermite nodes and applies adaptive
    numerical integration to the central position.  The latter is split at
    every ``q_finance(z) == q_river(z)`` boundary, which removes the previous
    discontinuous-indicator quadrature artefact.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.shape != (3,):
        raise ValueError("The adaptive central reference expects exactly three tokens")
    if system.landscape.dimension != 1:
        raise ValueError("The adaptive central reference is restricted to D=1")
    if context_order < 2:
        raise ValueError("context_order must be at least two")
    if not lower < upper:
        raise ValueError("Integration lower bound must be smaller than upper bound")

    landscape = system.landscape
    nodes, node_weights = np.polynomial.hermite.hermgauss(context_order)
    component_pairs = np.asarray(list(product(range(landscape.components), repeat=2)), dtype=np.int64)
    node_pairs = np.asarray(list(product(nodes, repeat=2)), dtype=np.float64)
    node_pair_weights = np.prod(
        np.asarray(list(product(node_weights / np.sqrt(np.pi), repeat=2)), dtype=np.float64), axis=1
    )
    context_weights: list[np.ndarray] = []
    context_left: list[np.ndarray] = []
    context_right: list[np.ndarray] = []
    for left_component, right_component in component_pairs:
        left_weight = landscape.weights(int(tokens[1]))[left_component]
        right_weight = landscape.weights(int(tokens[2]))[right_component]
        left = landscape.means[tokens[1], left_component, 0] + np.sqrt(2.0 * landscape.variances[tokens[1], left_component, 0]) * node_pairs[:, 0]
        right = landscape.means[tokens[2], right_component, 0] + np.sqrt(2.0 * landscape.variances[tokens[2], right_component, 0]) * node_pairs[:, 1]
        context_weights.append(float(left_weight * right_weight) * node_pair_weights)
        context_left.append(left[:, None])
        context_right.append(right[:, None])
    term_weights = np.concatenate(context_weights)
    left_states = np.concatenate(context_left)
    right_states = np.concatenate(context_right)
    left_right = system.kernel(left_states, right_states)

    def unnormalised_center_density(value: float) -> float:
        center = np.asarray([[value]], dtype=np.float64)
        center_left = system.kernel(center, left_states)
        center_right = system.kernel(center, right_states)
        exponent = system.coupling * (center_left + center_right + left_right) / 2.0
        total = float(np.sum(term_weights * np.exp(exponent)))
        density = float(np.exp(landscape.log_density(int(tokens[0]), center)).item())
        return density * total

    if boundaries is None:
        boundaries = _sense_boundaries_1d(q_finance, q_river, lower, upper, boundary_grid_size)
    elif any(not lower < boundary < upper for boundary in boundaries):
        raise ValueError("Every supplied boundary must lie inside the integration domain")
    intervals = (lower, *boundaries, upper)

    def region_credit(left: float, right: float) -> float:
        """Find a non-underflowing interior sign for one root-bounded interval."""
        for fraction in (0.5, 0.25, 0.75, 0.125, 0.875):
            value = left + fraction * (right - left)
            point = np.asarray([value], dtype=np.float64)
            finance = float(np.asarray(q_finance(point)).reshape(-1)[0])
            river = float(np.asarray(q_river(point)).reshape(-1)[0])
            if finance > river:
                return 1.0
            if finance < river:
                return 0.0
        return 0.5

    finance_mass = 0.0
    total_mass = 0.0
    error = 0.0
    for left, right in zip(intervals[:-1], intervals[1:]):
        mass, mass_error = quad(unnormalised_center_density, left, right, epsabs=epsabs, epsrel=epsrel, limit=200)
        credit = region_credit(left, right)
        finance_mass += credit * mass
        total_mass += mass
        error += abs(mass_error)
    if total_mass <= 0.0 or not np.isfinite(total_mass):
        raise RuntimeError("Adaptive central integration produced an invalid total mass")
    return AdaptiveCenterRegionMass(
        finance_mass=float(finance_mass),
        total_mass=float(total_mass),
        finance_probability=float(finance_mass / total_mass),
        integration_error=float(error / total_mass),
        boundaries=boundaries,
        lower=float(lower),
        upper=float(upper),
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

