"""Log existing lexical-capacity-ablation artifacts to the local MLflow ledger."""

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


def _diagnostic_metrics(result: dict[str, object]) -> dict[str, float]:
    windows = result["diagnosis"]["windows"]
    by_family = {
        family: [float(window["accuracy_credit_lambda1"]) for window in windows if window["family"] == family]
        for family in ("F", "R")
    }
    return {
        "validation.context_macro_accuracy_lambda1": sum(sum(values) / len(values) for values in by_family.values()) / 2.0,
        "validation.correct_region_gain_lambda1_minus_lambda0": sum(
            float(window["correct_region_gain"]) for window in windows
        )
        / len(windows),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=int, required=True, choices=(1, 4))
    parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args()
    artifact = PROJECT_ROOT / "experiments" / "runs" / f"gaussian-capacity-lexical-m{arguments.components}-seed{arguments.seed}.json"
    result = json.loads(artifact.read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("word-energy-gaussian-capacity-ablation")
    with mlflow.start_run(run_name=f"gaussian-lexical-capacity-m{arguments.components}-seed{arguments.seed}") as run:
        contract = RunContract(
            execution_tier="classical",
            device_class="cpu_gpu_simulator",
            model_family="normalised_d1_gaussian_mixture_analytic_overlap_capacity_ablation",
            energy_convention="compatibility C=-log integral p_w p_v; lambda=1 contextual diagnostic only",
            variable_encoding=f"continuous D=1 state; {arguments.components} Gaussian components per token",
            beta_temperature_convention="unit-temperature normalised free-energy ranking for diagnostic; no joint optimisation",
            sampler_schedule="analytic overlap plus adaptive 1-D diagnostic integration; no MCMC",
            seed=arguments.seed,
            data_hash=sha256_file(PROJECT_ROOT / "experiments" / "runs" / "synthetic-split-manifest.json"),
            data_split="synthetic train/validation; contextual test unopened",
            graph_summary="lexical training only; L=3 RBF system used only for validation diagnosis",
            limitations="Capacity ablation changes only lexical mixture components. It does not demonstrate joint-training benefit, sampler behaviour, natural-text performance or hardware execution.",
        )
        config = {**result["config"], "protocol_sha256": result["metrics"]["protocol.sha256"], "schedule_sha256": result["metrics"]["schedule.sha256"]}
        log_contract(mlflow, contract, config)
        audited_metrics = result.get("context_evaluation", {}).get("metrics", {})
        all_metrics = {**result["metrics"], **_diagnostic_metrics(result), **audited_metrics}
        mlflow.log_metrics({key: value for key, value in all_metrics.items() if isinstance(value, (int, float)) and math.isfinite(value)})
        mlflow.set_tags({key: str(value) for key, value in result["metrics"].items() if isinstance(value, str)})
        mlflow.log_artifact(str(artifact))
        checkpoint = artifact.with_suffix(".npz")
        if checkpoint.exists():
            mlflow.log_artifact(str(checkpoint))
        print(run.info.run_id)
