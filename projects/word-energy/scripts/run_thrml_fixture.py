"""Run the THRML simulator parity fixture; no physical-hardware claim."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from word_energy.ising.thrml_bridge import two_context_parity


if __name__ == "__main__":
    result = two_context_parity()
    destination = PROJECT_ROOT / "experiments" / "runs" / "thrml-six-spin-parity.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
