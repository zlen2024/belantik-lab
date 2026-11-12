"""THRML parity bridge for the hand-set six-spin Ising fixture.

The result is a THRML *simulator* measurement.  It intentionally contains no
claim that an Extropic physical backend executed the graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any
import math

import numpy as np


@dataclass(frozen=True)
class FixtureResult:
    context_bias: float
    exact_bank_plusplus: float
    exact_bank_minusminus: float
    sampled_bank_plusplus: float
    sampled_bank_minusminus: float
    samples: int
    warmup: int
    steps_per_sample: int
    seed: int
    execution_tier: str = "thrml_simulator"
    device_class: str = "cpu_gpu_simulator"


def _exact(context_bias: float) -> tuple[float, float]:
    rows: list[tuple[tuple[float, ...], float]] = []
    for spins in product((-1.0, 1.0), repeat=6):
        energy = -spins[0] * spins[1]
        energy -= context_bias * sum(spins[2:4])
        energy -= context_bias * sum(spins[4:6])
        energy -= 0.5 * sum(spins[index] * spins[2 + index] for index in range(2))
        energy -= 0.5 * sum(spins[index] * spins[4 + index] for index in range(2))
        rows.append((spins, math.exp(-energy)))
    normalizer = sum(weight for _, weight in rows)
    plus = sum(weight for spins, weight in rows if spins[:2] == (1.0, 1.0)) / normalizer
    minus = sum(weight for spins, weight in rows if spins[:2] == (-1.0, -1.0)) / normalizer
    return plus, minus


def sample_fixture(
    context_bias: float,
    *,
    warmup: int = 500,
    samples: int = 2000,
    steps_per_sample: int = 1,
    seed: int = 17,
) -> FixtureResult:
    """Sample the six-spin fixture with THRML and return parity-ready statistics."""
    import jax
    import jax.numpy as jnp
    from thrml import Block, SamplingSchedule, SpinNode, sample_states
    from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

    nodes = [SpinNode() for _ in range(6)]
    edges = [(nodes[0], nodes[1])]
    edges.extend((nodes[axis], nodes[offset + axis]) for offset in (2, 4) for axis in range(2))
    biases = jnp.array([0.0, 0.0, context_bias, context_bias, context_bias, context_bias])
    weights = jnp.array([1.0, 0.5, 0.5, 0.5, 0.5])
    model = IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))
    # This bipartition is valid for the five-edge fixture and enables parallel Gibbs updates.
    free_blocks = [Block([nodes[0], nodes[3], nodes[5]]), Block([nodes[1], nodes[2], nodes[4]])]
    program = IsingSamplingProgram(model, free_blocks, [])
    schedule = SamplingSchedule(n_warmup=warmup, n_samples=samples, steps_per_sample=steps_per_sample)
    key = jax.random.key(seed)
    key, init_key, sample_key = jax.random.split(key, 3)
    initial = hinton_init(init_key, model, free_blocks, ())
    observed = sample_states(sample_key, program, schedule, initial, [], [Block(nodes)])[0]
    spins = 2 * np.asarray(observed, dtype=np.int8) - 1
    bank = spins[:, :2]
    sampled_plus = float(np.mean(np.all(bank == 1, axis=1)))
    sampled_minus = float(np.mean(np.all(bank == -1, axis=1)))
    exact_plus, exact_minus = _exact(context_bias)
    return FixtureResult(
        context_bias=context_bias,
        exact_bank_plusplus=exact_plus,
        exact_bank_minusminus=exact_minus,
        sampled_bank_plusplus=sampled_plus,
        sampled_bank_minusminus=sampled_minus,
        samples=samples,
        warmup=warmup,
        steps_per_sample=steps_per_sample,
        seed=seed,
    )


def two_context_parity(**kwargs: Any) -> dict[str, dict[str, Any]]:
    return {
        "finance": asdict(sample_fixture(2.0, **kwargs)),
        "river": asdict(sample_fixture(-2.0, **kwargs)),
    }
