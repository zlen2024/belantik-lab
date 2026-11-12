"""Small continuous references used before MCMC or learned joint training."""

from __future__ import annotations

from itertools import product
import numpy as np

from word_energy.numerics import logsumexp

from .system import ActiveWindowSystem


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

