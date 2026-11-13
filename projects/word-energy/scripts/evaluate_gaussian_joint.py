"""Evaluate a saved Gaussian joint checkpoint on validation by default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from word_energy.energy_embedding.evaluate_joint_toy import evaluate_joint_context
from word_energy.energy_embedding.train_joint_toy import GaussianJointConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--test", action="store_true", help="Explicitly evaluate the held-out split after validation protocol is frozen.")
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--refinement-order", type=int, default=10)
    arguments = parser.parse_args()
    split = "test" if arguments.test else "validation"
    checkpoint = PROJECT_ROOT / "experiments" / "runs" / f"gaussian-joint-seed{arguments.seed}.npz"
    raw = dict(np.load(checkpoint))
    config = GaussianJointConfig(seed=arguments.seed)
    metrics = evaluate_joint_context(
        raw,
        config,
        split=split,
        order=arguments.order,
        refinement_order=arguments.refinement_order,
    )
    artifact = {
        "kind": "Gaussian joint train-anchor region evaluation; adaptive central integration and contextual quadrature audit required before semantic claim",
        "checkpoint": checkpoint.name,
        "split": split,
        "order": arguments.order,
        "refinement_order": arguments.refinement_order,
        "metrics": metrics,
    }
    target = PROJECT_ROOT / "experiments" / "runs" / f"gaussian-joint-{split}-evaluation-seed{arguments.seed}.json"
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))
