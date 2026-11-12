"""Run deterministic exact references for both Word Energy plans.

This writes a JSON artifact only.  It is a classical reference, not training,
not a THRML/Torx simulation, and not thermodynamic-chip execution.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from word_energy.energy_embedding import ActiveWindowSystem, GaussianMixtureLandscape, analytic_overlap
from word_energy.ising import IsingWordEnergy


def main() -> None:
    landscape = GaussianMixtureLandscape(
        logits=np.zeros((3, 2)),
        means=np.array([[[-2.0], [2.0]], [[2.0], [2.4]], [[-2.0], [-2.4]]]),
        variances=np.full((3, 2, 1), 0.25),
    )
    continuous = ActiveWindowSystem(landscape, raw_metric=np.array([[1.0]]), coupling=1.0)
    # A 31^3 grid is a fast smoke check.  The plan's refinement gates use
    # larger resolutions in a dedicated reference run before any training.
    grid = np.linspace(-8.0, 8.0, 31)
    from word_energy.energy_embedding.reference import grid_reference_1d

    log_z, readout = grid_reference_1d(continuous, np.array([0, 1, 2]), grid)
    ising = IsingWordEnergy(
        biases=np.array([[0.0, 0.0], [2.0, 2.0], [-2.0, -2.0]]),
        internal_couplings=np.array([1.0, 0.0, 0.0]),
        interaction_weights=np.array([0.5, 0.5]),
    )
    result = {
        "execution_tier": "classical",
        "device_class": "cpu_gpu_simulator",
        "continuous_gaussian_baseline": {
            "bank_finance_overlap": analytic_overlap(landscape, 0, 1),
            "bank_river_overlap": analytic_overlap(landscape, 0, 2),
            "joint_log_partition_grid": log_z,
            "joint_readout": readout.tolist(),
        },
        "ising_z1_primary": {
            "finance_bank_plusplus": ising.bank_region_mass(np.array([0, 1, 1]), True),
            "finance_bank_minusminus": ising.bank_region_mass(np.array([0, 1, 1]), False),
            "river_bank_plusplus": ising.bank_region_mass(np.array([0, 2, 2]), True),
            "river_bank_minusminus": ising.bank_region_mass(np.array([0, 2, 2]), False),
            "free_energy_lambda_zero": IsingWordEnergy(
                biases=ising.biases,
                internal_couplings=ising.internal_couplings,
                interaction_weights=ising.interaction_weights,
                coupling_scale=0.0,
            ).normalised_free_energy(np.array([0, 1, 1])),
        },
        "limitations": "Exact toy/reference only; no learned semantics, no THRML/Torx run, no hardware claim.",
    }
    destination = PROJECT_ROOT / "experiments" / "runs" / "reference-smoke.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
