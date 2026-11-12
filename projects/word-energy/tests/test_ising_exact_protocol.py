import numpy as np

from word_energy.ising.exact_training import prepare_data
from word_energy.synthetic import POOLS


def test_frozen_protocol_has_required_shapes_and_cross_pool_corruptions() -> None:
    data = prepare_data()
    assert data["train_pairs"].shape == (48 * 6, 2)
    assert data["train_negatives"].shape == (48 * 6, 5)
    assert data["train_corruptions"].shape == (48, 2, 3)
    assert data["lexical_schedule"].shape == (400, 64)
    assert data["joint_schedule"].shape == (100, 4)
    for window, corruptions, family in zip(
        data["train_windows"], data["train_corruptions"], data["train_families"]
    ):
        for corruption, slot in zip(corruptions, (1, 2)):
            assert np.array_equal(corruption[[0, 3 - slot]], window[[0, 3 - slot]])
            replacement = next(token for token, index in data["ids"].items() if index == corruption[slot])
            assert replacement not in POOLS[family]


def test_protocol_is_deterministic() -> None:
    first, second = prepare_data(), prepare_data()
    assert first["protocol_sha256"] == second["protocol_sha256"]
    assert first["schedule_sha256"] == second["schedule_sha256"]
