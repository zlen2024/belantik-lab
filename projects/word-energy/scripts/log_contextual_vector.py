"""Log contextual-vector validation or frozen holdout artifacts to local MLflow."""

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
    parser.add_argument("--variant", required=True, choices=("tied-d6", "untied-d3"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--test", action="store_true")
    arguments = parser.parse_args()
    split = "test" if arguments.test else "validation"
    source = PROJECT_ROOT / "experiments" / "runs" / (
        f"contextual-vector-{arguments.variant}-{split}-evaluation-seed{arguments.seed}.json"
        if arguments.test
        else f"contextual-vector-{arguments.variant}-seed{arguments.seed}.json"
    )
    result = json.loads(source.read_text(encoding="utf-8"))
    metrics = result["metrics"]
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("word-energy-contextual-vector")
    with mlflow.start_run(run_name=f"contextual-vector-{arguments.variant}-{split}-seed{arguments.seed}") as run:
        contract = RunContract(
            execution_tier="classical",
            device_class="cpu_gpu_simulator",
            model_family=f"SGNS_contextual_vector_{arguments.variant}",
            energy_convention="dot-product SGNS score; contextual readout h=u_bank+mean(u_context)",
            variable_encoding="continuous learned input/output embedding tables",
            beta_temperature_convention="not applicable",
            sampler_schedule="deterministic lexical embedding readout; no energy sampler",
            seed=arguments.seed,
            data_hash=sha256_file(PROJECT_ROOT / "experiments" / "runs" / "synthetic-split-manifest.json"),
            data_split=f"synthetic {split}; validation lexical selection; train-anchor centroids",
            graph_summary="no active joint graph; context is a mean vector",
            limitations="Matched contextual-vector control only. It cannot establish energy-model mechanisms, sampling fidelity, natural-text performance or hardware execution.",
        )
        config = result.get("config", {"variant": arguments.variant})
        log_contract(mlflow, contract, config)
        mlflow.log_metrics({key: value for key, value in metrics.items() if isinstance(value, (int, float)) and math.isfinite(value)})
        mlflow.set_tags({key: str(value) for key, value in metrics.items() if isinstance(value, str)})
        mlflow.log_artifact(str(source))
        checkpoint = source.with_name(f"contextual-vector-{arguments.variant}-seed{arguments.seed}.npz")
        if checkpoint.exists():
            mlflow.log_artifact(str(checkpoint))
        print(run.info.run_id)
