"""Attach adaptive numerical/context audits to completed lexical-capacity runs."""

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
    parser.add_argument("--components", type=int, nargs="+", choices=(1, 4), default=(1, 4))
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 29, 43))
    arguments = parser.parse_args()
    for components in arguments.components:
        for seed in arguments.seeds:
            target = PROJECT_ROOT / "experiments" / "runs" / f"gaussian-capacity-lexical-m{components}-seed{seed}.json"
            artifact = json.loads(target.read_text(encoding="utf-8"))
            checkpoint = target.with_suffix(".npz")
            metrics = evaluate_joint_context(
                dict(np.load(checkpoint)),
                GaussianJointConfig(seed=seed),
                split="validation",
                order=8,
                refinement_order=12,
            )
            artifact["context_evaluation"] = {
                "kind": "validation-only train-anchor adaptive region audit for frozen lexical geometry",
                "metrics": metrics,
            }
            target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
            print(json.dumps({"components": components, "seed": seed, "metrics": metrics}, indent=2))
