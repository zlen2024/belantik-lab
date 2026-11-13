"""Log a saved Gaussian joint validation evaluation to local MLflow."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow

from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--test", action="store_true")
    arguments = parser.parse_args()
    split = "test" if arguments.test else "validation"
    artifact = PROJECT_ROOT / "experiments" / "runs" / f"gaussian-joint-{split}-evaluation-seed{arguments.seed}.json"
    result = json.loads(artifact.read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("word-energy-gaussian-joint-quadrature")
    with mlflow.start_run(run_name=f"gaussian-joint-{split}-region-audit-seed{arguments.seed}") as run:
        contract = RunContract(
            execution_tier="classical",
            device_class="cpu_gpu_simulator",
            model_family="normalised_d1_m2_gaussian_mixture_joint_quadrature_region_audit",
            energy_convention="joint posterior region mass under train-anchor density map",
            variable_encoding="continuous D=1 Gaussian-mixture latent state",
            beta_temperature_convention="unit-temperature normalised free-energy ranking",
            sampler_schedule="adaptive 1-D central integration split at train-anchor region boundaries; Gauss--Hermite context quadrature; no MCMC",
            seed=arguments.seed,
            data_hash=sha256_file(PROJECT_ROOT / "experiments" / "runs" / "synthetic-split-manifest.json"),
            data_split=f"synthetic {split}; train-anchor map only",
            graph_summary="L=3 complete graph, bounded RBF kernel, D=1 fixed metric",
            limitations="Central-region discontinuity, context-node order and finite-domain sensitivity are measured on validation only. One seed is not a multi-seed result. No sampler or hardware claim.",
        )
        log_contract(mlflow, contract, result)
        mlflow.log_metrics({key: value for key, value in result["metrics"].items() if isinstance(value, (int, float)) and math.isfinite(value)})
        mlflow.log_artifact(str(artifact))
        print(run.info.run_id)
