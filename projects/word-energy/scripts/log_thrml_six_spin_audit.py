"""Log an existing four-chain THRML fixture audit to local MLflow."""

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
    arguments = parser.parse_args()
    artifact = PROJECT_ROOT / "experiments" / "runs" / f"thrml-six-spin-four-chain-audit-seed{arguments.seed}.json"
    result = json.loads(artifact.read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("word-energy-reference-foundation")
    with mlflow.start_run(run_name=f"ising-six-spin-thrml-four-chain-audit-seed{arguments.seed}") as run:
        finance = result["finance"]
        contract = RunContract(
            execution_tier="thrml_simulator",
            device_class="cpu_gpu_simulator",
            model_family="hand_set_six_spin_ising_fixture",
            energy_convention="E=-beta(sum b_i s_i + sum J_ij s_i s_j)",
            variable_encoding="Ising spins {-1,+1}; THRML bool True maps to +1",
            beta_temperature_convention="beta=1",
            sampler_schedule=(
                f"two-color Gibbs; {finance['chains']} dispersed chains; {finance['warmup']} warmup sweeps; "
                f"{finance['samples_per_chain']} retained sweeps/chain; {finance['steps_per_sample']} step per sample"
            ),
            chain_count=finance["chains"],
            burn_in=finance["warmup"],
            sample_count=finance["samples_per_chain"],
            seed=arguments.seed,
            data_hash=sha256_file(artifact),
            data_split="hand-set finance and river parity fixture; no learned corpus split",
            graph_summary="6 spins, 5 nonzero edges, maximum degree 3, valid two-color Gibbs blocks",
            limitations="THRML simulator only, hand-set fixture only; not trained language and not physical hardware.",
        )
        log_contract(mlflow, contract, result)
        metrics: dict[str, float] = {}
        for context in ("finance", "river"):
            for observable, values in result[context]["observables"].items():
                for name, value in values.items():
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        metrics[f"{context}.{observable}.{name}"] = value
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(artifact))
        print(run.info.run_id)
