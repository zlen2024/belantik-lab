"""Materialise the frozen synthetic split manifest for reproducible pilot runs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from word_energy.synthetic import manifest


if __name__ == "__main__":
    destination = PROJECT_ROOT / "experiments" / "runs" / "synthetic-split-manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest(), indent=2), encoding="utf-8")
    print(destination)
