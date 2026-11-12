"""Enumerate manual Ising relevance fixtures; no training or hardware claim."""
from itertools import product
from math import exp
from pathlib import Path
import json


def evaluate(context_biases):
    n = len(context_biases)
    total, scores = 0.0, [0.0] * n
    for spins in product((-1, 1), repeat=2 * (n + 1)):
        energy = -spins[0] * spins[1]
        energy -= sum(b * sum(spins[2 + 2*i:4 + 2*i]) for i, b in enumerate(context_biases))
        energy -= sum(spins[a] * spins[2 + 2*i + a] for i in range(n) for a in range(2)) / n
        weight = exp(-energy)
        total += weight
        for i in range(n):
            scores[i] += weight * sum(spins[a] * spins[2 + 2*i + a] for a in range(2)) / 2
    return [v / total for v in scores]


if __name__ == '__main__':
    conditions = {'ambiguous': [2, -2], 'dominant_finance': [2, 2, -2],
                  'dominant_river': [-2, -2, 2], 'swapped_ambiguous': [-2, 2]}
    result = {'kind': 'Exact hand-set distractor fixtures; not training or hardware',
              'conditions': {name: {'biases': c, 'readout': evaluate(c)} for name, c in conditions.items()}}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
