"""Validation-only diagnostics for paired Gaussian lexical/joint checkpoints."""

from __future__ import annotations

import numpy as np

from word_energy.energy_embedding.evaluate_joint_toy import _landscape_from_raw, _sense_map
from word_energy.energy_embedding.reference import _sense_boundaries_1d, adaptive_center_region_mass_1d
from word_energy.energy_embedding.system import ActiveWindowSystem
from word_energy.energy_embedding.train_joint_toy import GaussianJointConfig
from word_energy.ising.exact_training import prepare_data
from word_energy.synthetic import vocabulary


def _prediction(finance_probability: float) -> str:
    if finance_probability > 0.5:
        return "F"
    if finance_probability < 0.5:
        return "R"
    return "tie"


def _credit(prediction: str, family: str) -> float:
    return 1.0 if prediction == family else 0.5 if prediction == "tie" else 0.0


def diagnose_validation_windows(
    raw: dict[str, np.ndarray],
    config: GaussianJointConfig,
    *,
    order: int = 8,
) -> dict[str, object]:
    """Explain each validation F/R decision without touching the held-out test.

    The returned records use the same train-anchor map and lambda=0/lambda=1
    systems as the evaluator.  They are diagnostic evidence, not a checkpoint
    selector or a new metric.
    """
    data = prepare_data()
    names = vocabulary()
    landscape = _landscape_from_raw(raw, config)
    q_finance, q_river = _sense_map(landscape, data)
    boundaries = _sense_boundaries_1d(q_finance, q_river, -12.0, 12.0, grid_size=1025)
    coupled = ActiveWindowSystem(
        landscape, raw_metric=np.array([[1.0]]), coupling=config.coupling, length_scale=config.length_scale
    )
    uncoupled = ActiveWindowSystem(
        landscape, raw_metric=np.array([[1.0]]), coupling=0.0, length_scale=config.length_scale
    )
    records: list[dict[str, object]] = []
    for row, (window, family) in enumerate(zip(data["validation_windows"], data["validation_families"])):
        if family not in {"F", "R"}:
            continue
        posterior = adaptive_center_region_mass_1d(
            coupled, window, q_finance, q_river, context_order=order, boundaries=boundaries
        )
        prior = adaptive_center_region_mass_1d(
            uncoupled, window, q_finance, q_river, context_order=order, boundaries=boundaries
        )
        f_probability = posterior.finance_probability
        f_prior = prior.finance_probability
        true_probability = f_probability if family == "F" else 1.0 - f_probability
        true_prior = f_prior if family == "F" else 1.0 - f_prior
        prediction = _prediction(f_probability)
        records.append(
            {
                "validation_row": row,
                "family": family,
                "tokens": [names[int(token)] for token in window],
                "finance_probability_lambda0": f_prior,
                "finance_probability_lambda1": f_probability,
                "correct_region_probability_lambda0": true_prior,
                "correct_region_probability_lambda1": true_probability,
                "correct_region_gain": true_probability - true_prior,
                "prediction_lambda1": prediction,
                "accuracy_credit_lambda1": _credit(prediction, family),
            }
        )
    bank = names.index("bank")
    weights = landscape.weights(bank)
    bank_means = landscape.means[bank, :, 0]
    bank_variances = landscape.variances[bank, :, 0]
    return {
        "split": "validation_only",
        "order": order,
        "sense_boundaries": list(boundaries),
        "bank_landscape": {
            "component_weights": weights.tolist(),
            "component_means": bank_means.tolist(),
            "component_standard_deviations": np.sqrt(bank_variances).tolist(),
        },
        "windows": records,
    }
