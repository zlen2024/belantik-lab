"""Exact enumeration of a hand-set six-spin fixture; not training or chip code."""
from itertools import product
import json
import math
from pathlib import Path


def evaluate(context_bias):
    rows = []
    for spins in product((-1, 1), repeat=6):
        bank, context = spins[:2], (spins[2:4], spins[4:6])
        energy = -bank[0] * bank[1]
        energy -= sum(context_bias * sum(word) for word in context)
        energy -= 0.5 * sum(bank[d] * word[d] for word in context for d in range(2))
        rows.append((bank, math.exp(-energy)))
    partition = sum(weight for _, weight in rows)
    probability = lambda state: sum(w for b, w in rows if b == state) / partition
    plus, minus = probability((1, 1)), probability((-1, -1))
    return {"bank_plusplus": plus, "bank_minusminus": minus,
            "bank_other": 1.0 - plus - minus, "Z": partition}


if __name__ == "__main__":
    result = {"kind": "Analytic hand-set Ising toy; exact enumeration only; not training or hardware execution",
              "beta": 1, "bank_internal_coupling": 1, "context_bias_magnitude": 2,
              "interaction_coupling": 0.5, "n_spins": 6, "states": 64, "max_degree": 3,
              "finance": evaluate(2), "river": evaluate(-2)}
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
