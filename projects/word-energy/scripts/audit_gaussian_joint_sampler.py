"""Classical four-chain MH audit against the D=1 Gaussian exact reference."""

from __future__ import annotations

import argparse
import hashlib
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance(seed: int) -> dict[str, str]:
    checkpoint = PROJECT_ROOT / "experiments" / "runs" / f"gaussian-joint-seed{seed}.npz"
    return {
        "checkpoint_sha256": _sha256(checkpoint),
        "synthetic_split_manifest_sha256": _sha256(PROJECT_ROOT / "experiments" / "runs" / "synthetic-split-manifest.json"),
        "sampler_source_sha256": _sha256(PROJECT_ROOT / "word_energy" / "energy_embedding" / "sampler.py"),
        "reference_source_sha256": _sha256(PROJECT_ROOT / "word_energy" / "energy_embedding" / "reference.py"),
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
    }


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
    parser.add_argument(
        "--validation-rows",
        type=str,
        default=None,
        help="comma-separated validation rows for a bounded audit batch, for example 0,1",
    )
    parser.add_argument(
        "--combine-validation-batches",
        action="store_true",
        help="validate and combine completed two-row audit batches into the all-validation artifact",
    )
    arguments = parser.parse_args()
    selected_modes = sum((arguments.all_validation, arguments.validation_rows is not None, arguments.combine_validation_batches))
    if selected_modes > 1:
        parser.error("choose only one of --all-validation, --validation-rows, or --combine-validation-batches")

    if arguments.combine_validation_batches:
        data = prepare_data()
        expected_rows = list(range(len(data["validation_windows"])))
        batches: list[dict[str, object]] = []
        for start in range(0, len(expected_rows), 2):
            rows = expected_rows[start : start + 2]
            scope = "validation-rows-" + "-".join(str(index) for index in rows)
            source = PROJECT_ROOT / "experiments" / "runs" / f"gaussian-joint-sampler-audit-{scope}-seed{arguments.seed}.json"
            batches.append(json.loads(source.read_text(encoding="utf-8")))
        first = batches[0]
        for batch in batches:
            if (
                batch["seed"] != arguments.seed
                or batch["burn_in_sweeps"] != first["burn_in_sweeps"]
                or batch["retained_sweeps"] != first["retained_sweeps"]
            ):
                raise ValueError("all sampler audit batches must use the same seed, burn-in, and retained sweeps")
        windows = [window for batch in batches for window in batch["windows"]]
        if sorted(window["validation_row"] for window in windows) != expected_rows:
            raise ValueError("sampler audit batches do not cover each validation row exactly once")
        audit = {
            **first,
            "audit_scope": "all-validation-combined-from-batches",
            "provenance_at_combine": _provenance(arguments.seed),
            "windows": sorted(windows, key=lambda window: window["validation_row"]),
        }
        target = PROJECT_ROOT / "experiments" / "runs" / f"gaussian-joint-sampler-audit-all-validation-seed{arguments.seed}.json"
        target.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(json.dumps(audit, indent=2))
        raise SystemExit(0)

    config = GaussianJointConfig(seed=arguments.seed)
    raw = dict(np.load(PROJECT_ROOT / "experiments" / "runs" / f"gaussian-joint-seed{arguments.seed}.npz"))
    data = prepare_data()
    landscape = _landscape_from_raw(raw, config)
    q_finance, q_river = _sense_map(landscape, data)
    system = ActiveWindowSystem(landscape, raw_metric=np.array([[1.0]]), coupling=config.coupling, length_scale=config.length_scale)
    if arguments.validation_rows:
        selected_rows = tuple(int(value) for value in arguments.validation_rows.split(","))
        if not selected_rows or len(set(selected_rows)) != len(selected_rows):
            parser.error("--validation-rows must contain unique row numbers")
        if any(index < 0 or index >= len(data["validation_windows"]) for index in selected_rows):
            parser.error("--validation-rows contains a row outside the validation split")
        windows = [(index, data["validation_families"][index]) for index in selected_rows]
        scope = "validation-rows-" + "-".join(str(index) for index in selected_rows)
    elif arguments.all_validation:
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
        "provenance": _provenance(arguments.seed),
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
