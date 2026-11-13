"""Run a Gaussian joint quadrature precheck and save local artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from word_energy.energy_embedding.train_joint_toy import GaussianJointConfig, train_joint_precheck


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--smoke", action="store_true", help="Use 20 lexical and 10 joint updates for plumbing only.")
    arguments = parser.parse_args()
    config = (
        GaussianJointConfig(seed=arguments.seed, lexical_updates=20, joint_updates=10)
        if arguments.smoke
        else GaussianJointConfig(seed=arguments.seed)
    )
    raw, metrics = train_joint_precheck(config)
    artifact = {
        "kind": "Gaussian D=1/M=2 joint quadrature precheck; validation sense-first selection with adaptive region audit; no sampler, test or hardware claim",
        "config": config.__dict__,
        "metrics": metrics,
    }
    suffix = f"gaussian-joint-{'smoke-' if arguments.smoke else ''}seed{arguments.seed}.json"
    target = PROJECT_ROOT / "experiments" / "runs" / suffix
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    np.savez(target.with_suffix(".npz"), **raw)
    print(json.dumps(artifact, indent=2))
