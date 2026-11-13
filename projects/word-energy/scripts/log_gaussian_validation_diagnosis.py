"""Log the validation-only Gaussian geometry diagnosis to local MLflow."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow

from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri


def _macro_accuracy(windows: list[dict[str, object]]) -> float:
    scores = {
        family: [float(window["accuracy_credit_lambda1"]) for window in windows if window["family"] == family]
        for family in ("F", "R")
    }
    return sum(sum(values) / len(values) for values in scores.values()) / 2.0


if __name__ == "__main__":
    artifact = PROJECT_ROOT / "experiments" / "runs" / "gaussian-validation-geometry-diagnosis.json"
    result = json.loads(artifact.read_text(encoding="utf-8"))
    metrics: dict[str, float] = {}
    for seed, records in result["seeds"].items():
        lexical = records["lexical"]["diagnosis"]["windows"]
        joint = records["joint"]["diagnosis"]["windows"]
        metrics[f"seed{seed}.lexical_macro_accuracy_lambda1"] = _macro_accuracy(lexical)
        metrics[f"seed{seed}.joint_macro_accuracy_lambda1"] = _macro_accuracy(joint)
        metrics[f"seed{seed}.joint_minus_lexical_accuracy"] = (
            metrics[f"seed{seed}.joint_macro_accuracy_lambda1"] - metrics[f"seed{seed}.lexical_macro_accuracy_lambda1"]
        )
        metrics[f"seed{seed}.lexical_correct_region_gain_mean"] = sum(
            float(window["correct_region_gain"]) for window in lexical
        ) / len(lexical)
        metrics[f"seed{seed}.joint_correct_region_gain_mean"] = sum(
            float(window["correct_region_gain"]) for window in joint
        ) / len(joint)
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("word-energy-gaussian-joint-quadrature")
    with mlflow.start_run(run_name="gaussian-validation-lexical-joint-geometry-diagnosis") as run:
        contract = RunContract(
            execution_tier="classical",
            device_class="cpu_gpu_simulator",
            model_family="paired_d1_m2_gaussian_lexical_and_joint_validation_geometry_diagnosis",
            energy_convention="train-anchor region occupancy at lambda=0 and lambda=1",
            variable_encoding="continuous D=1 Gaussian-mixture latent state",
            beta_temperature_convention="unit-temperature normalised free-energy ranking",
            sampler_schedule="adaptive 1-D central integration with Gauss--Hermite context quadrature; no MCMC",
            seed=0,
            data_hash=sha256_file(PROJECT_ROOT / "experiments" / "runs" / "synthetic-split-manifest.json"),
            data_split="synthetic validation only; train-anchor maps; contextual test unopened",
            graph_summary="L=3 complete graph, bounded RBF kernel, D=1 fixed metric",
            limitations="Post-hoc diagnosis of completed validation runs. It identifies seed-43 lexical-region bias but does not test a new hypothesis, sampler or hardware mapping.",
        )
        log_contract(mlflow, contract, {"artifact_kind": result["kind"], "order": result["order"], "seeds": list(result["seeds"])})
        mlflow.log_metrics({key: value for key, value in metrics.items() if math.isfinite(value)})
        mlflow.log_artifact(str(artifact))
        print(run.info.run_id)
