"""Evaluate a frozen contextual-vector checkpoint; validation is the default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from word_energy.contextual_vector import ContextualVectorConfig, evaluate_contextual_vector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("tied-d6", "untied-d3"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--test", action="store_true", help="Explicitly evaluate the frozen holdout after its manifest is committed.")
    arguments = parser.parse_args()
    tied, dimension = (True, 6) if arguments.variant == "tied-d6" else (False, 3)
    config = ContextualVectorConfig(seed=arguments.seed, tied=tied, dimension=dimension)
    checkpoint = PROJECT_ROOT / "experiments" / "runs" / f"contextual-vector-{arguments.variant}-seed{arguments.seed}.npz"
    split = "test" if arguments.test else "validation"
    metrics = evaluate_contextual_vector(dict(np.load(checkpoint)), config, split)
    artifact = {
        "kind": "frozen contextual-vector SGNS evaluation; no energy dynamics or hardware claim",
        "checkpoint": checkpoint.name,
        "variant": arguments.variant,
        "split": split,
        "metrics": metrics,
    }
    target = PROJECT_ROOT / "experiments" / "runs" / f"contextual-vector-{arguments.variant}-{split}-evaluation-seed{arguments.seed}.json"
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))
