"""Oracle anchored parity with two hand-set target unary minima."""
from itertools import product
from pathlib import Path
import json
import sys

import mlflow
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_two_cue_ising_fixture import brute_force
from word_energy.tracking import RunContract, log_contract, tracking_uri
from word_energy.wikitext_four_spin_wic import STATES, posterior, unary


def main():
    params = {'biases': np.zeros((7, 4)), 'internal': np.zeros((7, 6)),
              'interaction': np.full(4, np.log(.8/.7))}
    # 0 target; 1,2 A cues; 3,4 B cues; 5 and 6 class anchors.
    for token, dim, sign in ((1, 0, -1), (2, 0, 1), (3, 1, -1), (4, 1, 1)):
        params['biases'][token, dim] = np.arctanh(sign*2/2.5)
    for token in (0, 5, 6):
        params['biases'][token, 2:] = np.arctanh(.2/2.5)
    params['internal'][0, 0] = np.arctanh(.1/2)
    params['internal'][5, 0] = np.arctanh(-1/2)
    params['internal'][6, 0] = np.arctanh(1/2)
    q = unary(params)
    coupling = 1.5*expit(params['interaction'])
    region = (q[5] > q[6]).astype(float) + .5*(q[5] == q[6])
    minima = []
    for state in range(len(STATES)):
        neighbors = [state ^ (1 << spin) for spin in range(4)]
        if np.all(q[0, state] > q[0, neighbors]):
            minima.append(state)
    expected_minima = [i for i, s in enumerate(STATES) if s[0] == s[1] and s[2] == 1 and s[3] == 1]
    if minima != expected_minima or len(minima) != 2:
        raise ValueError('target minima differ from protocol')
    prior = float(q[0] @ region)
    # c=-1,u=2,v=2 gives 3/4; the additive upper bound follows from
    # equal sums of diagonal and off-diagonal scores.
    additive_scores = [-1, 1, 1, 3]
    additive_accuracy = np.mean([(s > 0) == bool(y) for s, y in zip(additive_scores, (0, 1, 1, 0))])
    cases = []
    for a, b in product((0, 1), repeat=2):
        cue_a, cue_b = 1+a, 3+b
        actual = posterior(q, coupling, 0, [cue_a, cue_b])
        exact = brute_force(q, coupling, 0, cue_a, cue_b)
        np.testing.assert_allclose(actual, exact, atol=1e-12, rtol=0)
        np.testing.assert_allclose(actual, posterior(q, coupling, 0, [cue_b, cue_a]), atol=1e-12, rtol=0)
        np.testing.assert_allclose(posterior(q, np.zeros(4), 0, [cue_a, cue_b]), q[0], atol=1e-12, rtol=0)
        mass = float(actual @ region)
        label = a ^ b
        product_prediction = (-(2*a-1)*(2*b-1)) > 0
        if product_prediction != bool(label):
            raise ValueError('nonlinear vector product fixture mismatch')
        credit = .5 if mass == .5 else float((mass > .5) == bool(label))
        single_a = float(posterior(q, coupling/2, 0, [cue_a]) @ region)
        single_b = float(posterior(q, coupling/2, 0, [cue_b]) @ region)
        cases.append({'a': a, 'b': b, 'label': label, 'region_probability': mass, 'credit': credit,
                      'single_a_probability': single_a, 'single_b_probability': single_b,
                      'max_enumeration_error': float(np.max(np.abs(actual-exact)))})
    output = {'target_minima_indices': minima, 'target_minima_states': STATES[minima].tolist(),
              'anchor1_region_indices': np.flatnonzero(region == 1).tolist(),
              'anchor_tie_indices': np.flatnonzero(region == .5).tolist(), 'prior_region_probability': prior,
              'cases': cases, 'accuracy': float(np.mean([row['credit'] for row in cases])),
              'additive_accuracy_witness': float(additive_accuracy), 'nonlinear_product_accuracy': 1.,
              'minimum_abs_margin': min(abs(row['region_probability']-.5) for row in cases),
              'parameters': {k: v.tolist() for k, v in params.items()},
              'limitations': 'hand-set target, anchors and cues; oracle anchor family labels; no language training, heldout sense evaluation or physical hardware'}
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment('word-energy-two-cue-fixtures')
    with mlflow.start_run(run_name='ising-hand-set-two-minima-anchor-map') as run:
        log_contract(mlflow, RunContract('classical', 'cpu_gpu_simulator', 'ising_two_cue_anchored_fixture',
            'Ising spin logits with hand-set pair couplings and fields', 'spins {-1,+1}', 'beta=1',
            'exact production posterior and 4096-state enumeration', 0, 'hand-constructed-anchored-fixture-v1',
            'synthetic known-answer fixture', '12 spins, 26 edges', output['limitations']), output)
        mlflow.log_dict(output, 'fixture.json')
        for path in (Path(__file__), ROOT/'scripts/audit_two_cue_ising_fixture.py',
                     ROOT/'notes/TWO_CUE_ANCHORED_FIXTURE_PROTOCOL.md', ROOT/'word_energy/wikitext_four_spin_wic.py'):
            mlflow.log_artifact(str(path), 'source_snapshot')
        print('RUN', run.info.run_id, json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
