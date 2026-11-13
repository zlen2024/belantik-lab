"""Train-anchor sense-map evaluation for the Gaussian joint toy precheck."""

from __future__ import annotations

import numpy as np

from word_energy.energy_embedding.landscape import GaussianMixtureLandscape
from word_energy.energy_embedding.reference import _sense_boundaries_1d, adaptive_center_region_mass_1d
from word_energy.energy_embedding.system import ActiveWindowSystem
from word_energy.energy_embedding.train_joint_toy import GaussianJointConfig
from word_energy.energy_embedding.train_lexical import GaussianLexicalConfig, _constrained, _imports
from word_energy.ising.exact_training import prepare_data


def _landscape_from_raw(raw: dict[str, np.ndarray], config: GaussianJointConfig) -> GaussianMixtureLandscape:
    _, jnp, _ = _imports()
    components = int(np.asarray(raw["logits"]).shape[1])
    lexical_config = GaussianLexicalConfig(
        seed=config.seed,
        updates=config.lexical_updates,
        batch_size=config.lexical_batch,
        learning_rate=config.lexical_lr,
        clip_norm=config.clip_norm,
        validation_every=config.validation_every,
        components=components,
    )
    parameters = _constrained({name: jnp.asarray(value) for name, value in raw.items()}, lexical_config, jnp)
    return GaussianMixtureLandscape(
        logits=np.asarray(parameters["logits"]),
        means=np.asarray(parameters["means"]),
        variances=np.asarray(parameters["variances"]),
    )


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
    lower: float = -12.0,
    upper: float = 12.0,
    refinement_lower: float = -15.0,
    refinement_upper: float = 15.0,
) -> dict[str, float]:
    """Evaluate context sense mass with train-only regions and adaptive integration.

    ``split='test'`` is explicit to protect the holdout.  The returned
    refinement quantities separately audit the context Gauss--Hermite nodes
    and finite integration domain.  Callers must report them before
    interpreting a sense score.
    """
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    if refinement_order <= order:
        raise ValueError("refinement_order must exceed order")
    data = prepare_data()
    landscape = _landscape_from_raw(raw, config)
    q_finance, q_river = _sense_map(landscape, data)
    boundaries = _sense_boundaries_1d(q_finance, q_river, lower, upper, grid_size=1025)
    refinement_boundaries = _sense_boundaries_1d(
        q_finance, q_river, refinement_lower, refinement_upper, grid_size=1025
    )
    system = ActiveWindowSystem(landscape, raw_metric=np.array([[1.0]]), coupling=config.coupling, length_scale=config.length_scale)
    lambda_zero_system = ActiveWindowSystem(landscape, raw_metric=np.array([[1.0]]), coupling=0.0, length_scale=config.length_scale)
    credits: dict[str, list[float]] = {"F": [], "R": []}
    prior_credits: dict[str, list[float]] = {"F": [], "R": []}
    counts: dict[str, dict[str, int]] = {"F": {"F": 0, "R": 0, "tie": 0}, "R": {"F": 0, "R": 0, "tie": 0}}
    posterior: list[float] = []
    prior: list[float] = []
    context_refinement_deltas: list[float] = []
    domain_refinement_deltas: list[float] = []
    integration_errors: list[float] = []
    for window, family in zip(data[f"{split}_windows"], data[f"{split}_families"]):
        if family not in credits:
            continue
        marginal = adaptive_center_region_mass_1d(
            system, window, q_finance, q_river, context_order=order, lower=lower, upper=upper, boundaries=boundaries
        )
        zero_marginal = adaptive_center_region_mass_1d(
            lambda_zero_system,
            window,
            q_finance,
            q_river,
            context_order=order,
            lower=lower,
            upper=upper,
            boundaries=boundaries,
        )
        refined_context = adaptive_center_region_mass_1d(
            system,
            window,
            q_finance,
            q_river,
            context_order=refinement_order,
            lower=lower,
            upper=upper,
            boundaries=boundaries,
        )
        refined_domain = adaptive_center_region_mass_1d(
            system,
            window,
            q_finance,
            q_river,
            context_order=refinement_order,
            lower=refinement_lower,
            upper=refinement_upper,
            boundaries=refinement_boundaries,
        )
        f_mass = marginal.finance_probability
        f_prior = zero_marginal.finance_probability
        true_mass = f_mass if family == "F" else 1.0 - f_mass
        true_prior = f_prior if family == "F" else 1.0 - f_prior
        prediction = "F" if f_mass > 0.5 else "R" if f_mass < 0.5 else "tie"
        counts[family][prediction] += 1
        credits[family].append(1.0 if true_mass > 0.5 else 0.5 if true_mass == 0.5 else 0.0)
        prior_credits[family].append(1.0 if true_prior > 0.5 else 0.5 if true_prior == 0.5 else 0.0)
        posterior.append(true_mass)
        prior.append(true_prior)
        context_refinement_deltas.append(abs(f_mass - refined_context.finance_probability))
        domain_refinement_deltas.append(abs(refined_context.finance_probability - refined_domain.finance_probability))
        integration_errors.extend(
            [marginal.integration_error, zero_marginal.integration_error, refined_context.integration_error, refined_domain.integration_error]
        )
    max_context_delta = float(np.max(context_refinement_deltas))
    max_domain_delta = float(np.max(domain_refinement_deltas))
    return {
        f"{split}.macro_sense_accuracy": float(np.mean([np.mean(credits["F"]), np.mean(credits["R"])])),
        f"{split}.lambda1_macro_sense_accuracy": float(np.mean([np.mean(credits["F"]), np.mean(credits["R"])])),
        f"{split}.lambda0_macro_sense_accuracy": float(
            np.mean([np.mean(prior_credits["F"]), np.mean(prior_credits["R"])])
        ),
        f"{split}.context_f_correct_count": float(counts["F"]["F"]),
        f"{split}.context_r_correct_count": float(counts["R"]["R"]),
        f"{split}.context_f_predicted_r": float(counts["F"]["R"]),
        f"{split}.context_r_predicted_f": float(counts["R"]["F"]),
        f"{split}.context_tie_count": float(counts["F"]["tie"] + counts["R"]["tie"]),
        f"{split}.correct_region_posterior": float(np.mean(posterior)),
        f"{split}.correct_region_prior": float(np.mean(prior)),
        f"{split}.correct_region_gain_lambda1_minus_lambda0": float(np.mean(posterior) - np.mean(prior)),
        f"{split}.region_quadrature_order": float(order),
        f"{split}.region_refinement_order": float(refinement_order),
        f"{split}.region_max_context_order_delta": max_context_delta,
        f"{split}.region_max_domain_delta": max_domain_delta,
        f"{split}.region_max_adaptive_integration_error": float(np.max(integration_errors)),
        f"{split}.region_max_mass_delta": max(max_context_delta, max_domain_delta),
    }
