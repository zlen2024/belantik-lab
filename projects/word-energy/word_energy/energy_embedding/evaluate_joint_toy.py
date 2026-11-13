"""Train-anchor sense-map evaluation for the Gaussian joint toy precheck."""

from __future__ import annotations

import numpy as np

from word_energy.energy_embedding.landscape import GaussianMixtureLandscape
from word_energy.energy_embedding.reference import gaussian_hermite_center_marginal_1d
from word_energy.energy_embedding.system import ActiveWindowSystem
from word_energy.energy_embedding.train_joint_toy import GaussianJointConfig
from word_energy.energy_embedding.train_lexical import GaussianLexicalConfig, _constrained, _imports
from word_energy.ising.exact_training import prepare_data


def _landscape_from_raw(raw: dict[str, np.ndarray], config: GaussianJointConfig) -> GaussianMixtureLandscape:
    _, jnp, _ = _imports()
    lexical_config = GaussianLexicalConfig(
        seed=config.seed,
        updates=config.lexical_updates,
        batch_size=config.lexical_batch,
        learning_rate=config.lexical_lr,
        clip_norm=config.clip_norm,
        validation_every=config.validation_every,
    )
    parameters = _constrained({name: jnp.asarray(value) for name, value in raw.items()}, lexical_config, jnp)
    return GaussianMixtureLandscape(
        logits=np.asarray(parameters["logits"]),
        means=np.asarray(parameters["means"]),
        variances=np.asarray(parameters["variances"]),
    )


def _region_mass(marginal, q_finance, q_river) -> float:
    difference = q_finance(marginal.states) - q_river(marginal.states)
    f_mass = float(np.sum(marginal.weights[difference > 0]))
    equality = np.isclose(difference, 0.0, rtol=0.0, atol=1e-14)
    return f_mass + 0.5 * float(np.sum(marginal.weights[equality]))


def _sense_map(landscape: GaussianMixtureLandscape, data):
    anchors: dict[str, list[int]] = {"F": [], "R": []}
    for window, family in zip(data["train_windows"], data["train_families"]):
        if family in anchors:
            anchors[family].extend(window[1:].tolist())

    def density(family: str, positions: np.ndarray) -> np.ndarray:
        points = np.asarray(positions, dtype=np.float64).reshape(-1, 1)
        return np.mean([np.exp(landscape.log_density(token, points)) for token in anchors[family]], axis=0)

    return lambda positions: density("F", positions), lambda positions: density("R", positions)


def evaluate_joint_context(
    raw: dict[str, np.ndarray],
    config: GaussianJointConfig = GaussianJointConfig(),
    *,
    split: str = "validation",
    order: int = 12,
    refinement_order: int = 16,
) -> dict[str, float]:
    """Evaluate context sense mass with a train-only density-derived region map.

    ``split='test'`` is explicit to protect the holdout.  The returned
    refinement quantity measures only the discontinuous region integral's
    sensitivity to quadrature order; callers must report it before interpreting
    a sense score.
    """
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    if refinement_order <= order:
        raise ValueError("refinement_order must exceed order")
    data = prepare_data()
    landscape = _landscape_from_raw(raw, config)
    q_finance, q_river = _sense_map(landscape, data)
    system = ActiveWindowSystem(landscape, raw_metric=np.array([[1.0]]), coupling=config.coupling, length_scale=config.length_scale)
    lambda_zero_system = ActiveWindowSystem(landscape, raw_metric=np.array([[1.0]]), coupling=0.0, length_scale=config.length_scale)
    credits: dict[str, list[float]] = {"F": [], "R": []}
    posterior: list[float] = []
    prior: list[float] = []
    refinement_deltas: list[float] = []
    for window, family in zip(data[f"{split}_windows"], data[f"{split}_families"]):
        if family not in credits:
            continue
        marginal = gaussian_hermite_center_marginal_1d(system, window, order=order)
        zero_marginal = gaussian_hermite_center_marginal_1d(lambda_zero_system, window, order=order)
        refined = gaussian_hermite_center_marginal_1d(system, window, order=refinement_order)
        f_mass = _region_mass(marginal, q_finance, q_river)
        f_prior = _region_mass(zero_marginal, q_finance, q_river)
        refined_f_mass = _region_mass(refined, q_finance, q_river)
        true_mass = f_mass if family == "F" else 1.0 - f_mass
        true_prior = f_prior if family == "F" else 1.0 - f_prior
        credits[family].append(1.0 if true_mass > 0.5 else 0.5 if true_mass == 0.5 else 0.0)
        posterior.append(true_mass)
        prior.append(true_prior)
        refinement_deltas.append(abs(f_mass - refined_f_mass))
    return {
        f"{split}.macro_sense_accuracy": float(np.mean([np.mean(credits["F"]), np.mean(credits["R"])])),
        f"{split}.correct_region_posterior": float(np.mean(posterior)),
        f"{split}.correct_region_prior": float(np.mean(prior)),
        f"{split}.correct_region_gain_lambda1_minus_lambda0": float(np.mean(posterior) - np.mean(prior)),
        f"{split}.region_quadrature_order": float(order),
        f"{split}.region_refinement_order": float(refinement_order),
        f"{split}.region_max_mass_delta": float(np.max(refinement_deltas)),
    }
