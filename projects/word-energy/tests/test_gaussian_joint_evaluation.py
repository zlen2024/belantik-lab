import numpy as np

from word_energy.energy_embedding.evaluate_joint_toy import evaluate_joint_context
from word_energy.energy_embedding.train_joint_toy import GaussianJointConfig
from word_energy.synthetic import vocabulary


def test_gaussian_joint_context_evaluation_uses_explicit_validation_split() -> None:
    vocab_size = len(vocabulary())
    raw = {
        "means": np.zeros((vocab_size, 2, 1), dtype=np.float64),
        "raw_variances": np.zeros((vocab_size, 2, 1), dtype=np.float64),
        "logits": np.zeros((vocab_size, 2), dtype=np.float64),
    }
    metrics = evaluate_joint_context(
        raw,
        GaussianJointConfig(quadrature_order=4),
        split="validation",
        order=4,
        refinement_order=6,
    )
    assert 0.0 <= metrics["validation.macro_sense_accuracy"] <= 1.0
    assert 0.0 <= metrics["validation.correct_region_posterior"] <= 1.0
    assert metrics["validation.region_max_context_order_delta"] <= 1.0
    assert metrics["validation.region_max_domain_delta"] <= 1.0
    assert metrics["validation.region_max_adaptive_integration_error"] <= 1.0
    assert metrics["validation.region_max_mass_delta"] <= 1.0
