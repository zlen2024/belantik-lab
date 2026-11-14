"""Run a four-chain THRML simulator audit on the known-answer Ising fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from word_energy.ising.thrml_bridge import two_context_four_chain_audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=2_000)
    parser.add_argument("--steps-per-sample", type=int, default=1)
    arguments = parser.parse_args()
    result = two_context_four_chain_audit(
        seed=arguments.seed,
        warmup=arguments.warmup,
        samples=arguments.samples,
        steps_per_sample=arguments.steps_per_sample,
    )
    destination = PROJECT_ROOT / "experiments" / "runs" / f"thrml-six-spin-four-chain-audit-seed{arguments.seed}.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
