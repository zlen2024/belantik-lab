"""Run the preregistered D=1 lexical mixture-capacity ablation on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from word_energy.energy_embedding.diagnostics import diagnose_validation_windows
from word_energy.energy_embedding.train_joint_toy import GaussianJointConfig
from word_energy.energy_embedding.train_lexical import GaussianLexicalConfig, train_lexical


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=int, nargs="+", choices=(1, 4), default=(1, 4))
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 29, 43))
    arguments = parser.parse_args()
    for components in arguments.components:
        for seed in arguments.seeds:
            config = GaussianLexicalConfig(seed=seed, components=components)
            raw, metrics = train_lexical(config)
            diagnosis = diagnose_validation_windows(raw, GaussianJointConfig(seed=seed), order=8)
            artifact = {
                "kind": "preregistered D=1 Gaussian lexical mixture-capacity ablation; validation-only contextual diagnosis; no joint retraining or test",
                "config": config.__dict__,
                "metrics": metrics,
                "diagnosis": diagnosis,
            }
            target = PROJECT_ROOT / "experiments" / "runs" / f"gaussian-capacity-lexical-m{components}-seed{seed}.json"
            target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
            np.savez(target.with_suffix(".npz"), **raw)
            print(json.dumps({"components": components, "seed": seed, "metrics": metrics}, indent=2))
