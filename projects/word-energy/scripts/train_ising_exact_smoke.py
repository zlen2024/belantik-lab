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
    arguments = parser.parse_args()
    config = (
        ExactTrainingConfig(seed=arguments.seed)
        if arguments.full
        else ExactTrainingConfig(seed=arguments.seed, lexical_updates=20, joint_updates=10)
    )
    raw, metrics = train_smoke(config)
    if arguments.evaluate_test:
        metrics.update(evaluate_checkpoint(raw, split="test"))
    stage = "one-seed exact-gradient Ising frozen-budget precheck" if arguments.full else "one-seed exact-gradient Ising smoke"
    artifact = {"kind": f"{stage}; not sampler training or hardware", "config": config.__dict__, "metrics": metrics}
    suffix = f"ising-exact-training-seed{arguments.seed}.json" if arguments.full else f"ising-exact-training-smoke-seed{arguments.seed}.json"
    target = PROJECT_ROOT / "experiments" / "runs" / suffix
    np.savez(target.with_suffix(".npz"), **raw)
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))
