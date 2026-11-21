"""Hand-constructed parity fixture on the production four-spin star."""
from itertools import product
from pathlib import Path
import json
import sys

import mlflow
import numpy as np
from scipy.special import expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from word_energy.tracking import RunContract, log_contract, tracking_uri
from word_energy.wikitext_four_spin_wic import STATES, posterior, unary


def brute_force(q, coupling, target, cue_a, cue_b):
    """Enumerate all three 16-state variables independently."""
    terms = (np.log(q[target])[:, None, None] + np.log(q[cue_a])[None, :, None]
             + np.log(q[cue_b])[None, None, :])
    cross = (STATES * (coupling/2)) @ STATES.T
    terms = terms + cross[:, :, None] + cross[:, None, :]
    marginal = np.exp(logsumexp(terms, axis=(1, 2)) - logsumexp(terms))
    return marginal


def main():
    params = {'biases': np.zeros((5, 4)), 'internal': np.zeros((5, 6)),
              'interaction': np.full(4, np.log(.8/.7))}
    for token, dim, sign in ((1, 0, -1), (2, 0, 1), (3, 1, -1), (4, 1, 1)):
        params['biases'][token, dim] = np.arctanh(sign*2/2.5)
    q = unary(params)
    coupling = 1.5*expit(params['interaction'])
    region = STATES[:, 0]*STATES[:, 1] < 0
    cases = []
    for a, b in product((0, 1), repeat=2):
        cue_a, cue_b = 1+a, 3+b
        exact = brute_force(q, coupling, 0, cue_a, cue_b)
        actual = posterior(q, coupling, 0, [cue_a, cue_b])
        np.testing.assert_allclose(actual, exact, atol=1e-12, rtol=0)
        np.testing.assert_allclose(actual, posterior(q, coupling, 0, [cue_b, cue_a]), atol=1e-12, rtol=0)
        assert abs(posterior(q, np.zeros(4), 0, [cue_a, cue_b])[region].sum()-.5) < 1e-12
        mass = float(actual[region].sum())
        label = a ^ b
        assert (mass > .5) == bool(label) and abs(mass-.5) > 1e-10
        vector_product = -(2*a-1)*(2*b-1)
        assert (vector_product > 0) == bool(label)
        cases.append({'a': a, 'b': b, 'label': label, 'region_probability': mass,
                      'max_enumeration_error': float(np.max(np.abs(actual-exact)))})
    # A linear threshold has scores c, c+u, c+v, c+u+v. If the two
    # off-diagonal cells are positive and diagonal cells negative, adding
    # the first/last inequalities contradicts adding the middle two.
    # This small enumeration supplies a concrete 3/4 witness, not the proof.
    best = 0
    for c, u, v in product(range(-3, 4), repeat=3):
        scores = [c, c+v, c+u, c+u+v]
        if 0 in scores:
            continue
        best = max(best, sum((score > 0) == bool(label) for score, label in zip(scores, (0, 1, 1, 0))))
    assert best == 3
    output = {'cases': cases, 'minimum_abs_margin': min(abs(c['region_probability']-.5) for c in cases),
              'max_additive_accuracy_witness': best/4, 'nonlinear_product_accuracy': 1.,
              'parameters': {k: v.tolist() for k,v in params.items()},
              'limitations': 'hand-designed discrete symbols, region and parameters; no training, heldout sense task or physical hardware'}
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment('word-energy-two-cue-fixtures')
    with mlflow.start_run(run_name='ising-hand-constructed-parity') as run:
        log_contract(mlflow, RunContract('classical', 'cpu_gpu_simulator', 'ising_two_cue_parity_fixture',
            'Ising spin logits and positive cross couplings', 'spins {-1,+1}', 'beta=1',
            'exact production factorization and 4096-state enumeration', 0, 'hand-constructed-fixture-v1',
            'synthetic known-answer fixture', '12 spins, 26 edges', output['limitations']), output)
        mlflow.log_dict(output, 'fixture.json')
        for path in (Path(__file__), ROOT/'notes/TWO_CUE_ISING_FIXTURE_PROTOCOL.md', ROOT/'word_energy/wikitext_four_spin_wic.py'):
            mlflow.log_artifact(str(path), 'source_snapshot')
        print('RUN', run.info.run_id, json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
