"""Log a completed classical Gaussian sampler audit to local MLflow."""

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
    parser.add_argument(
        "--all-validation",
        action="store_true",
        help="log the audit artifact generated with --all-validation",
    )
    arguments = parser.parse_args()
    scope = "all-validation" if arguments.all_validation else "two-fixture"
    artifact = PROJECT_ROOT / "experiments" / "runs" / f"gaussian-joint-sampler-audit-{scope}-seed{arguments.seed}.json"
    result = json.loads(artifact.read_text(encoding="utf-8"))
    metrics: dict[str, float] = {}
    for window in result["windows"]:
        prefix = f"validation_row{window['validation_row']}"
        metrics[f"{prefix}.acceptance_rate"] = window["acceptance_rate"]
        for name, values in window["observables"].items():
            for metric, value in values.items():
                metrics[f"{prefix}.{name}.{metric}"] = value
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("word-energy-gaussian-joint-quadrature")
    with mlflow.start_run(run_name=f"gaussian-joint-classical-sampler-audit-{scope}-seed{arguments.seed}") as run:
        contract = RunContract(
            execution_tier="classical",
            device_class="cpu_gpu_simulator",
            model_family="normalised_d1_m2_gaussian_joint_independence_metropolis_hastings",
            energy_convention="dimensionless beta-E; unary independence proposals cancel from MH ratio",
            variable_encoding="continuous D=1 Gaussian-mixture latent state",
            beta_temperature_convention="unit temperature; lambda=1 bounded RBF interaction",
            sampler_schedule=f"4 chains; {result['burn_in_sweeps']} burn-in sweeps; {result['retained_sweeps']} retained sweeps; no thinning",
            seed=arguments.seed,
            data_hash=sha256_file(PROJECT_ROOT / "experiments" / "runs" / "synthetic-split-manifest.json"),
            data_split=(
                "all fixed validation windows; contextual holdout not used for sampler tuning"
                if arguments.all_validation
                else "two fixed validation windows; contextual holdout not used for sampler tuning"
            ),
            graph_summary="L=3 complete graph, unary independence MH proposals, random site order",
            limitations=(
                "Classical MCMC audit across the validation split only; not THRML/Torx or physical hardware."
                if arguments.all_validation
                else "Initial sampler fixture only: two validation windows and one model seed. Classical MCMC, not THRML/Torx or physical hardware."
            ),
        )
        log_contract(mlflow, contract, result)
        mlflow.log_metrics({key: value for key, value in metrics.items() if isinstance(value, (int, float)) and math.isfinite(value)})
        mlflow.log_artifact(str(artifact))
        print(run.info.run_id)
