"""Joint independence-Metropolis sampler for the small continuous Gaussian reference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .system import ActiveWindowSystem


@dataclass(frozen=True)
class JointSamplerResult:
    """Retained joint states and proposal accounting for a dimensionless beta-E model."""

    samples: np.ndarray  # [chains, sweeps, window, dimension]
    acceptance_rate: float
    accepted: int
    proposed: int


def _unary_proposal(system: ActiveWindowSystem, token: int, rng: np.random.Generator) -> np.ndarray:
    landscape = system.landscape
    component = int(rng.choice(landscape.components, p=landscape.weights(token)))
    return landscape.means[token, component] + rng.normal(size=landscape.dimension) * np.sqrt(
        landscape.variances[token, component]
    )


def dispersed_unary_starts(system: ActiveWindowSystem, tokens: np.ndarray, chains: int, rng: np.random.Generator) -> np.ndarray:
    """Start chains in cyclic, distinct mixture components when capacity permits."""
    tokens = np.asarray(tokens, dtype=np.int64)
    states = np.empty((chains, len(tokens), system.landscape.dimension), dtype=np.float64)
    for chain in range(chains):
        component = chain % system.landscape.components
        for position, token in enumerate(tokens):
            states[chain, position] = system.landscape.means[token, component] + rng.normal(
                size=system.landscape.dimension
            ) * np.sqrt(system.landscape.variances[token, component])
    return states


def sample_joint_independence_mh(
    system: ActiveWindowSystem,
    tokens: np.ndarray,
    *,
    chains: int = 4,
    burn_in_sweeps: int = 500,
    sample_sweeps: int = 2_000,
    seed: int = 0,
) -> JointSamplerResult:
    """Sample one shared active window using unary independence proposals.

    Proposal ``z_i' ~ p_{w_i}`` cancels its unary factor in the MH ratio.  For
    the joint energy ``sum U - lambda*pair_sum/(L-1)``, the log acceptance is
    therefore ``lambda*(pair_sum_new-pair_sum_old)/(L-1)``.  This is a
    classical reference sampler, not THRML/Torx or physical hardware.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.ndim != 1 or len(tokens) < 2:
        raise ValueError("Joint MH requires at least two token IDs")
    if chains < 2 or burn_in_sweeps < 0 or sample_sweeps < 1:
        raise ValueError("Need at least two chains, non-negative burn-in and positive retained sweeps")
    rng = np.random.default_rng(seed)
    states = dispersed_unary_starts(system, tokens, chains, rng)
    retained = np.empty((chains, sample_sweeps, len(tokens), system.landscape.dimension), dtype=np.float64)
    accepted = proposed = 0
    for sweep in range(burn_in_sweeps + sample_sweeps):
        for chain in range(chains):
            for position in rng.permutation(len(tokens)):
                current = states[chain, position].copy()
                proposal = _unary_proposal(system, int(tokens[position]), rng)
                others = [other for other in range(len(tokens)) if other != position]
                old_pairs = sum(system.kernel(current, states[chain, other]) for other in others)
                new_pairs = sum(system.kernel(proposal, states[chain, other]) for other in others)
                log_acceptance = system.coupling * (new_pairs - old_pairs) / (len(tokens) - 1)
                proposed += 1
                if np.log(rng.random()) < min(0.0, float(log_acceptance)):
                    states[chain, position] = proposal
                    accepted += 1
        if sweep >= burn_in_sweeps:
            retained[:, sweep - burn_in_sweeps] = states
    return JointSamplerResult(
        samples=retained,
        acceptance_rate=accepted / proposed,
        accepted=accepted,
        proposed=proposed,
    )
