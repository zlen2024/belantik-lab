"""Run one Gaussian lexical-baseline precheck and write local artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from word_energy.energy_embedding.train_lexical import GaussianLexicalConfig, evaluate_lexical, train_lexical


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--evaluate-test", action="store_true")
    arguments = parser.parse_args()
    config = GaussianLexicalConfig(seed=arguments.seed, updates=arguments.updates)
    raw, metrics = train_lexical(config)
    if arguments.evaluate_test:
        metrics.update(evaluate_lexical(raw, "test"))
    artifact = {
        "kind": "Gaussian-mixture lexical baseline; analytic overlap only; no joint equilibrium or hardware claim",
        "config": config.__dict__,
        "metrics": metrics,
    }
    target = PROJECT_ROOT / "experiments" / "runs" / f"gaussian-lexical-seed{arguments.seed}.json"
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    np.savez(target.with_suffix(".npz"), **raw)
    print(json.dumps(artifact, indent=2))
