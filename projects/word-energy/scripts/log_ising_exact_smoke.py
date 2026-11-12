"""Log the exact-gradient Ising smoke artifact using the local MLflow contract."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow

from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Log the frozen-budget one-seed precheck artifact.")
    parser.add_argument("--seed", type=int, default=17, help="Model initialisation seed to log.")
    parser.add_argument("--random-baseline", action="store_true", help="Log the untrained initialisation control artifact.")
    arguments = parser.parse_args()
    suffix = (
        f"ising-random-seed{arguments.seed}.json"
        if arguments.random_baseline
        else f"ising-exact-training-seed{arguments.seed}.json"
        if arguments.full
        else f"ising-exact-training-smoke-seed{arguments.seed}.json"
    )
    artifact = PROJECT_ROOT / "experiments" / "runs" / suffix
    result = json.loads(artifact.read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("word-energy-exact-training")
    stage = "random-control" if arguments.random_baseline else "precheck" if arguments.full else "smoke"
    run_name = f"ising-exact-ranking-{stage}-seed{arguments.seed}"
    with mlflow.start_run(run_name=run_name) as run:
        contract = RunContract(
            execution_tier="classical",
            device_class="cpu_gpu_simulator",
            model_family="two_spin_per_word_exact_free_energy_ranking",
            energy_convention="normalised F2/F3 free-energy ranking; beta=1",
            variable_encoding="Ising spins {-1,+1}",
            beta_temperature_convention="beta=1",
            sampler_schedule="exact enumeration of 4 unary and 64 joint states; no sampler gradient",
            seed=result["config"]["seed"],
            data_hash=sha256_file(PROJECT_ROOT / "experiments" / "runs" / "synthetic-split-manifest.json"),
            data_split="synthetic protocol with held-out test evaluation; random control has no checkpoint selection",
            graph_summary="L=3 star, 2 spins per token, 7 potential edges, maximum degree 3",
            limitations="Synthetic control/precheck only; no sampler audit, physical-hardware execution or natural-text result.",
        )
        config = {**result["config"], "protocol_sha256": result["metrics"]["protocol.sha256"], "schedule_sha256": result["metrics"]["schedule.sha256"]}
        log_contract(mlflow, contract, config)
        mlflow.log_metrics(
            {key: value for key, value in result["metrics"].items() if isinstance(value, (int, float)) and math.isfinite(value)}
        )
        mlflow.set_tags({key: str(value) for key, value in result["metrics"].items() if isinstance(value, str)})
        mlflow.log_artifact(str(artifact))
        checkpoint = artifact.with_suffix(".npz")
        if checkpoint.exists():
            mlflow.log_artifact(str(checkpoint))
        print(run.info.run_id)
