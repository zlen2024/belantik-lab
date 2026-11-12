"""Run one exact-gradient Ising toy smoke and persist a JSON artifact."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import argparse
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from word_energy.ising.exact_training import ExactTrainingConfig, evaluate_checkpoint, train_smoke


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Use the frozen 400+100 update budget for one seed.")
    parser.add_argument("--seed", type=int, default=17, help="Model initialisation seed; protocol data stay fixed.")
    parser.add_argument("--evaluate-test", action="store_true", help="Open the held-out test split after the selected checkpoint is fixed.")
    parser.add_argument("--random-baseline", action="store_true", help="Evaluate the untrained initialisation control with the same readout.")
    parser.add_argument("--permuted-lexical-control", action="store_true", help="Permute lexical positive contexts with frozen RNG 8080.")
    arguments = parser.parse_args()
    if arguments.permuted_lexical_control and not arguments.full:
        parser.error("--permuted-lexical-control requires --full")
    if arguments.permuted_lexical_control and arguments.random_baseline:
        parser.error("choose either --random-baseline or --permuted-lexical-control")
    if arguments.random_baseline:
        config = ExactTrainingConfig(seed=arguments.seed, lexical_updates=0, joint_updates=0)
    elif arguments.full:
        config = ExactTrainingConfig(seed=arguments.seed, permuted_lexical_contexts=arguments.permuted_lexical_control)
    else:
        config = ExactTrainingConfig(seed=arguments.seed, lexical_updates=20, joint_updates=10)
    raw, metrics = train_smoke(config)
    if arguments.random_baseline:
        metrics["training.stage"] = "untrained random-initialisation control"
    if arguments.evaluate_test:
        metrics.update(evaluate_checkpoint(raw, split="test"))
    stage = (
        "untrained Ising random-initialisation control"
        if arguments.random_baseline
        else "one-seed Ising frozen lexical-context permutation control"
        if arguments.permuted_lexical_control
        else "one-seed exact-gradient Ising frozen-budget precheck"
        if arguments.full
        else "one-seed exact-gradient Ising smoke"
    )
    artifact = {"kind": f"{stage}; not sampler training or hardware", "config": config.__dict__, "metrics": metrics}
    suffix = (
        f"ising-random-seed{arguments.seed}.json"
        if arguments.random_baseline
        else f"ising-permuted-lexical-seed{arguments.seed}.json"
        if arguments.permuted_lexical_control
        else f"ising-exact-training-seed{arguments.seed}.json"
        if arguments.full
        else f"ising-exact-training-smoke-seed{arguments.seed}.json"
    )
    target = PROJECT_ROOT / "experiments" / "runs" / suffix
    np.savez(target.with_suffix(".npz"), **raw)
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))
