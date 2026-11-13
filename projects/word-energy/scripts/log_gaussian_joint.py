"""Log an existing Gaussian joint quadrature artifact to local MLflow."""

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
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    suffix = f"gaussian-joint-{'smoke-' if arguments.smoke else ''}seed{arguments.seed}.json"
    artifact = PROJECT_ROOT / "experiments" / "runs" / suffix
    result = json.loads(artifact.read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("word-energy-gaussian-joint-quadrature")
    stage = "smoke" if arguments.smoke else "precheck"
    with mlflow.start_run(run_name=f"gaussian-joint-quadrature-{stage}-seed{arguments.seed}") as run:
        contract = RunContract(
            execution_tier="classical",
            device_class="cpu_gpu_simulator",
            model_family="normalised_d1_m2_gaussian_mixture_joint_quadrature",
            energy_convention="F3=-log E_product_unary[exp(lambda pair-kernel sum)]; beta=1",
            variable_encoding="continuous D=1 Gaussian-mixture latent state",
            beta_temperature_convention="unit-temperature normalised free-energy ranking",
            sampler_schedule="exact-for-fixed-order Gauss--Hermite quadrature; no sampler gradient",
            seed=result["config"]["seed"],
            data_hash=sha256_file(PROJECT_ROOT / "experiments" / "runs" / "synthetic-split-manifest.json"),
            data_split="synthetic train with validation sense-first/loss-second checkpoint selection; test unopened",
            graph_summary="L=3 complete graph, bounded RBF kernel, D=1 fixed metric",
            limitations="Validation selector includes adaptive sense-region audit, but this remains a numerical precheck: no sampler audit, test evaluation, multi-seed result or hardware execution.",
        )
        config = {**result["config"], "protocol_sha256": result["metrics"]["protocol.sha256"], "schedule_sha256": result["metrics"]["schedule.sha256"]}
        log_contract(mlflow, contract, config)
        mlflow.log_metrics({key: value for key, value in result["metrics"].items() if isinstance(value, (int, float)) and math.isfinite(value)})
        mlflow.set_tags({key: str(value) for key, value in result["metrics"].items() if isinstance(value, str)})
        mlflow.log_artifact(str(artifact))
        checkpoint = artifact.with_suffix(".npz")
        if checkpoint.exists():
            mlflow.log_artifact(str(checkpoint))
        print(run.info.run_id)
