"""Log completed Word Energy reference artifacts to the explicit local MLflow store."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow

from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri


ARTIFACT_DIR = PROJECT_ROOT / "experiments" / "runs"


def _read(name: str) -> tuple[Path, dict]:
    path = ARTIFACT_DIR / name
    return path, json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("word-energy-reference-foundation")

    classical_path, classical = _read("reference-smoke.json")
    with mlflow.start_run(run_name="gaussian-and-ising-classical-reference") as run:
        contract = RunContract(
            execution_tier="classical",
            device_class="cpu_gpu_simulator",
            model_family="gaussian_mixture_baseline_and_exact_ising",
            energy_convention="normalised unary -log p; dimensionless beta=1 Ising energy",
            variable_encoding="continuous R^1 and Ising spins {-1,+1}",
            beta_temperature_convention="beta=1",
            sampler_schedule="exact enumeration; coarse 31^3 grid smoke only",
            seed=17,
            data_hash=sha256_file(classical_path),
            data_split="hand-set reference fixture",
            graph_summary="continuous L=3 all-pairs; Ising L=3 star with 7 potential edges",
            limitations=classical["limitations"],
        )
        log_contract(mlflow, contract, {"artifact": classical_path.name, "grid_points_per_axis": 31})
        mlflow.log_metric("ising.finance.bank_plusplus", classical["ising_z1_primary"]["finance_bank_plusplus"])
        mlflow.log_metric("ising.river.bank_minusminus", classical["ising_z1_primary"]["river_bank_minusminus"])
        mlflow.log_metric("ising.free_energy_lambda_zero", classical["ising_z1_primary"]["free_energy_lambda_zero"])
        mlflow.log_artifact(str(classical_path))
        print(f"classical_run_id={run.info.run_id}")

    thrml_path, thrml = _read("thrml-six-spin-parity.json")
    with mlflow.start_run(run_name="ising-six-spin-thrml-simulator-parity") as run:
        finance = thrml["finance"]
        river = thrml["river"]
        contract = RunContract(
            execution_tier="thrml_simulator",
            device_class="cpu_gpu_simulator",
            model_family="hand_set_six_spin_ising_fixture",
            energy_convention="E=-beta(sum b_i s_i + sum J_ij s_i s_j)",
            variable_encoding="Ising spins {-1,+1}; THRML bool True maps to +1",
            beta_temperature_convention="beta=1",
            sampler_schedule="chromatic Gibbs, two color blocks",
            chain_count=1,
            burn_in=finance["warmup"],
            sample_count=finance["samples"],
            seed=finance["seed"],
            data_hash=sha256_file(thrml_path),
            data_split="hand-set parity fixture",
            graph_summary="6 spins, 5 nonzero edges, maximum degree 3",
            limitations="One-chain smoke parity only; ESS/R-hat and four-chain audit remain required before a simulation gate claim.",
        )
        log_contract(mlflow, contract, {"artifact": thrml_path.name, "contexts": ["finance", "river"]})
        for name, payload in (("finance", finance), ("river", river)):
            mlflow.log_metric(f"{name}.abs_error.bank_plusplus", abs(payload["sampled_bank_plusplus"] - payload["exact_bank_plusplus"]))
            mlflow.log_metric(f"{name}.abs_error.bank_minusminus", abs(payload["sampled_bank_minusminus"] - payload["exact_bank_minusminus"]))
        mlflow.log_artifact(str(thrml_path))
        print(f"thrml_run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
