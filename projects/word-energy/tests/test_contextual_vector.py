import numpy as np

from word_energy.contextual_vector import ContextualVectorConfig, evaluate_contextual_vector
from word_energy.synthetic import vocabulary


def test_contextual_vector_readout_uses_explicit_validation_and_reports_counts() -> None:
    config = ContextualVectorConfig()
    raw = {"input": np.zeros((len(vocabulary()), config.dimension), dtype=np.float64)}
    metrics = evaluate_contextual_vector(raw, config, "validation")
    assert metrics["validation.context_macro_sense_accuracy"] == 0.5
    assert metrics["validation.context_tie_count"] == 12.0
