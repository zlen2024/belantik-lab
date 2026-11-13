"""Train frozen-budget tied and untied contextual-vector controls on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from word_energy.contextual_vector import ContextualVectorConfig, evaluate_contextual_vector, train_contextual_vector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", choices=("tied-d6", "untied-d3"), nargs="+", default=("tied-d6", "untied-d3"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 29, 43))
    arguments = parser.parse_args()
    for variant in arguments.variants:
        tied, dimension = (True, 6) if variant == "tied-d6" else (False, 3)
        for seed in arguments.seeds:
            config = ContextualVectorConfig(seed=seed, tied=tied, dimension=dimension)
            raw, metrics = train_contextual_vector(config)
            metrics.update(evaluate_contextual_vector(raw, config, "validation"))
            artifact = {
                "kind": "frozen-budget contextual SGNS baseline; validation-only checkpoint selection and contextual readout; test unopened",
                "config": config.__dict__,
                "metrics": metrics,
            }
            target = PROJECT_ROOT / "experiments" / "runs" / f"contextual-vector-{variant}-seed{seed}.json"
            target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
            np.savez(target.with_suffix(".npz"), **raw)
            print(json.dumps({"variant": variant, "seed": seed, "metrics": metrics}, indent=2))
