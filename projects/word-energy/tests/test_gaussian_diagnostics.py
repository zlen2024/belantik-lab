import numpy as np

from word_energy.energy_embedding.diagnostics import diagnose_validation_windows
from word_energy.energy_embedding.train_joint_toy import GaussianJointConfig
from word_energy.synthetic import vocabulary


def test_gaussian_validation_diagnosis_is_validation_only_and_window_resolved() -> None:
    vocab_size = len(vocabulary())
    raw = {
        "means": np.zeros((vocab_size, 2, 1), dtype=np.float64),
        "raw_variances": np.zeros((vocab_size, 2, 1), dtype=np.float64),
        "logits": np.zeros((vocab_size, 2), dtype=np.float64),
    }
    diagnosis = diagnose_validation_windows(raw, GaussianJointConfig(), order=4)
    assert diagnosis["split"] == "validation_only"
    assert len(diagnosis["windows"]) == 12
    assert {window["family"] for window in diagnosis["windows"]} == {"F", "R"}
    assert all("test" not in str(window) for window in diagnosis["windows"])
