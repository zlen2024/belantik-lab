"""Exact small-graph implementation of the frozen Ising pilot contract."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np

from word_energy.numerics import logsumexp


def all_spin_states(spins: int) -> np.ndarray:
    return np.asarray(list(product((-1.0, 1.0), repeat=spins)), dtype=np.float64)


@dataclass(frozen=True)
class IsingWordEnergy:
    """Two spin states per word type, with a star-shaped active window.

    The model follows the Z1-first pilot: occurrence 0 is the central token,
    all occurrences share type parameters, and inter-token coupling is only
    between corresponding spin coordinates.
    """

    biases: np.ndarray  # [vocabulary, 2]
    internal_couplings: np.ndarray  # [vocabulary]
    interaction_weights: np.ndarray  # [2]
    beta: float = 1.0
    coupling_scale: float = 1.0

    def __post_init__(self) -> None:
        biases = np.asarray(self.biases, dtype=np.float64)
        internal = np.asarray(self.internal_couplings, dtype=np.float64)
        interactions = np.asarray(self.interaction_weights, dtype=np.float64)
        if biases.ndim != 2 or biases.shape[1] != 2:
            raise ValueError("biases must have shape [vocabulary, 2]")
        if internal.shape != (biases.shape[0],) or interactions.shape != (2,):
            raise ValueError("Ising parameter shapes disagree")
        if self.beta <= 0 or self.coupling_scale < 0:
            raise ValueError("beta must be positive and coupling scale non-negative")
        object.__setattr__(self, "biases", biases)
        object.__setattr__(self, "internal_couplings", internal)
        object.__setattr__(self, "interaction_weights", interactions)

    def unary_energy(self, token: int, spins: np.ndarray) -> np.ndarray:
        spins = np.asarray(spins, dtype=np.float64)
        return -self.beta * (
            np.einsum("...a,a->...", spins, self.biases[token])
            + self.internal_couplings[token] * spins[..., 0] * spins[..., 1]
        )

    def unary_log_partition(self, token: int) -> float:
        return float(logsumexp(-self.unary_energy(token, all_spin_states(2))))

    def physical_energy(self, tokens: np.ndarray, states: np.ndarray) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.int64)
        states = np.asarray(states, dtype=np.float64)
        if states.shape[-2:] != (len(tokens), 2):
            raise ValueError("Expected states[..., window, 2]")
        energy = np.zeros(states.shape[:-2], dtype=np.float64)
        for position, token in enumerate(tokens):
            energy += self.unary_energy(int(token), states[..., position, :])
        for position in range(1, len(tokens)):
            energy -= self.beta * self.coupling_scale * np.sum(
                self.interaction_weights * states[..., 0, :] * states[..., position, :], axis=-1
            )
        return energy

    def exact_distribution(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tokens = np.asarray(tokens, dtype=np.int64)
        states = all_spin_states(2 * len(tokens)).reshape(-1, len(tokens), 2)
        energies = self.physical_energy(tokens, states)
        log_probabilities = -energies - logsumexp(-energies)
        return states, np.exp(log_probabilities)

    def normalised_free_energy(self, tokens: np.ndarray) -> float:
        tokens = np.asarray(tokens, dtype=np.int64)
        _, probabilities = self.exact_distribution(tokens)
        # log Z_phys is recovered without exposing an arbitrary energy offset.
        states = all_spin_states(2 * len(tokens)).reshape(-1, len(tokens), 2)
        log_z_joint = float(logsumexp(-self.physical_energy(tokens, states)))
        return float(-log_z_joint + sum(self.unary_log_partition(int(token)) for token in tokens))

    def joint_readout(self, tokens: np.ndarray) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.int64)
        states, probabilities = self.exact_distribution(tokens)
        result = np.eye(len(tokens), dtype=np.float64)
        for left in range(len(tokens)):
            for right in range(left + 1, len(tokens)):
                score = np.mean(states[:, left, :] * states[:, right, :], axis=1)
                result[left, right] = result[right, left] = float(np.dot(probabilities, score))
        return result

    def bank_region_mass(self, tokens: np.ndarray, positive_region: bool) -> float:
        states, probabilities = self.exact_distribution(tokens)
        bank = states[:, 0, :]
        mask = np.all(bank == (1.0 if positive_region else -1.0), axis=1)
        return float(np.sum(probabilities[mask]))

