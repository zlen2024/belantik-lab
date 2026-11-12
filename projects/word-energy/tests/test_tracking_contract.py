from pathlib import Path

import pytest

from word_energy.tracking import RunContract, tracking_uri


def test_word_energy_uses_an_explicit_local_sqlite_uri() -> None:
    uri = tracking_uri(Path(__file__).resolve().parents[1])
    assert uri.startswith("sqlite:///")
    assert uri.endswith("/tracking/mlflow.db")


def test_hardware_tier_requires_a_physical_chip() -> None:
    with pytest.raises(ValueError):
        RunContract(
            execution_tier="hardware",
            device_class="remote_simulator",
            model_family="toy",
            energy_convention="dimensionless",
            variable_encoding="spin",
            beta_temperature_convention="beta=1",
            sampler_schedule="exact",
            seed=0,
            data_hash="none",
            data_split="synthetic",
            graph_summary="none",
            limitations="test",
        )
