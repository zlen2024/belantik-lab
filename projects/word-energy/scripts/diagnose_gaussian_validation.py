"""Write a validation-only comparison of Gaussian lexical and joint checkpoints."""

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 29, 43))
    parser.add_argument("--order", type=int, default=8)
    arguments = parser.parse_args()
    comparison: dict[str, object] = {
        "kind": "validation-only Gaussian lexical-versus-joint geometry diagnosis; no test evaluation",
        "order": arguments.order,
        "seeds": {},
    }
    for seed in arguments.seeds:
        config = GaussianJointConfig(seed=seed)
        records: dict[str, object] = {}
        for checkpoint_kind in ("lexical", "joint"):
            checkpoint = PROJECT_ROOT / "experiments" / "runs" / f"gaussian-{checkpoint_kind}-seed{seed}.npz"
            records[checkpoint_kind] = {
                "checkpoint": checkpoint.name,
                "diagnosis": diagnose_validation_windows(dict(np.load(checkpoint)), config, order=arguments.order),
            }
        comparison["seeds"][str(seed)] = records
    target = PROJECT_ROOT / "experiments" / "runs" / "gaussian-validation-geometry-diagnosis.json"
    target.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
