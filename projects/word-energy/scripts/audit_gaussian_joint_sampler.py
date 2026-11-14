"""Classical four-chain MH audit against the D=1 Gaussian exact reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from word_energy.energy_embedding.evaluate_joint_toy import _landscape_from_raw, _sense_map
from word_energy.energy_embedding.mcmc_diagnostics import split_rank_rhat_ess
from word_energy.energy_embedding.reference import adaptive_center_region_mass_1d, gaussian_hermite_joint_reference
from word_energy.energy_embedding.sampler import sample_joint_independence_mh
from word_energy.energy_embedding.system import ActiveWindowSystem
from word_energy.energy_embedding.train_joint_toy import GaussianJointConfig
from word_energy.ising.exact_training import prepare_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--samples", type=int, default=2_000)
    parser.add_argument(
        "--all-validation",
        action="store_true",
        help="audit every validation window; default remains one F and one R fixture",
    )
    arguments = parser.parse_args()
    config = GaussianJointConfig(seed=arguments.seed)
    raw = dict(np.load(PROJECT_ROOT / "experiments" / "runs" / f"gaussian-joint-seed{arguments.seed}.npz"))
    data = prepare_data()
    landscape = _landscape_from_raw(raw, config)
    q_finance, q_river = _sense_map(landscape, data)
    system = ActiveWindowSystem(landscape, raw_metric=np.array([[1.0]]), coupling=config.coupling, length_scale=config.length_scale)
    if arguments.all_validation:
        windows = list(enumerate(data["validation_families"]))
        scope = "all-validation"
    else:
        windows = [(0, "F"), (6, "R")]
        scope = "two-fixture"
    audit: dict[str, object] = {
        "kind": "classical four-chain independence-MH audit against Gaussian quadrature reference; no hardware claim",
        "seed": arguments.seed,
        "audit_scope": scope,
        "burn_in_sweeps": arguments.burn_in,
        "retained_sweeps": arguments.samples,
        "windows": [],
    }
    for index, family in windows:
        tokens = data["validation_windows"][index]
        result = sample_joint_independence_mh(
            system, tokens, chains=4, burn_in_sweeps=arguments.burn_in, sample_sweeps=arguments.samples, seed=10_000 + index
        )
        samples = result.samples
        exact = gaussian_hermite_joint_reference(system, tokens, order=12)
        exact_basin = adaptive_center_region_mass_1d(system, tokens, q_finance, q_river, context_order=12).finance_probability
        observables: dict[str, object] = {}
        center_states = samples[:, :, 0, 0]
        finance_density = q_finance(center_states.reshape(-1)).reshape(center_states.shape)
        river_density = q_river(center_states.reshape(-1)).reshape(center_states.shape)
        basin = (finance_density > river_density).astype(np.float64)
        rhat, ess, mcse = split_rank_rhat_ess(basin)
        empirical = float(np.mean(basin))
        observables["finance_basin"] = {
            "empirical": empirical,
            "reference": exact_basin,
            "absolute_error": abs(empirical - exact_basin),
            "rhat": rhat,
            "ess": ess,
            "mcse": mcse,
        }
        for left in range(3):
            for right in range(left + 1, 3):
                values = system.kernel(samples[:, :, left], samples[:, :, right])
                rhat, ess, mcse = split_rank_rhat_ess(values)
                empirical = float(np.mean(values))
                observables[f"R_{left}{right}"] = {
                    "empirical": empirical,
                    "reference": float(exact.readout[left, right]),
                    "absolute_error": abs(empirical - float(exact.readout[left, right])),
                    "rhat": rhat,
                    "ess": ess,
                    "mcse": mcse,
                }
        audit["windows"].append(
            {"validation_row": index, "family": family, "tokens": tokens.tolist(), "acceptance_rate": result.acceptance_rate, "observables": observables}
        )
    target = PROJECT_ROOT / "experiments" / "runs" / f"gaussian-joint-sampler-audit-{scope}-seed{arguments.seed}.json"
    target.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
