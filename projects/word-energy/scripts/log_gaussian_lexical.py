"""Log an existing Gaussian lexical-baseline artifact to local MLflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow

from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args()
    artifact = PROJECT_ROOT / "experiments" / "runs" / f"gaussian-lexical-seed{arguments.seed}.json"
    result = json.loads(artifact.read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("word-energy-gaussian-lexical")
    with mlflow.start_run(run_name=f"gaussian-lexical-precheck-seed{arguments.seed}") as run:
        contract = RunContract(
            execution_tier="classical",
            device_class="cpu_gpu_simulator",
            model_family="normalised_diagonal_gaussian_mixture_analytic_overlap",
            energy_convention="compatibility C=-log integral p_w p_v; lexical ranking only",
            variable_encoding="continuous D=1 state; two Gaussian components per token",
            beta_temperature_convention="not applicable; no joint equilibrium in this lexical baseline",
            sampler_schedule="analytic Gaussian-mixture overlap; no sampling and no sampler gradient",
            seed=result["config"]["seed"],
            data_hash=sha256_file(PROJECT_ROOT / "experiments" / "runs" / "synthetic-split-manifest.json"),
            data_split="synthetic train with frozen validation checkpoint selection; held-out lexical test opened",
            graph_summary="no active joint graph in lexical-only baseline",
            limitations="Lexical baseline only: no Gaussian joint free energy, context sense readout, sampler audit, hardware mapping or natural-text result.",
        )
        config = {**result["config"], "protocol_sha256": result["metrics"]["protocol.sha256"], "schedule_sha256": result["metrics"]["schedule.sha256"]}
        log_contract(mlflow, contract, config)
        mlflow.log_metrics({key: value for key, value in result["metrics"].items() if isinstance(value, (int, float))})
        mlflow.set_tags({key: str(value) for key, value in result["metrics"].items() if isinstance(value, str)})
        mlflow.log_artifact(str(artifact))
        checkpoint = artifact.with_suffix(".npz")
        if checkpoint.exists():
            mlflow.log_artifact(str(checkpoint))
        print(run.info.run_id)
