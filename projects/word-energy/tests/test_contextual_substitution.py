"""Independent numerical and invariance checks for contextual readouts."""
import numpy as np
import pytest
from scipy.special import expit

from word_energy.contextual_substitution import ising_candidates, ising_context_only_candidates, ising_kernel_candidates, log_bhattacharyya, vector_candidates
from word_energy.wikitext_four_spin_wic import STATES, unary


def fixture():
    rng = np.random.default_rng(7)
    return {'biases': rng.normal(0, .2, (5, 4)), 'internal': rng.normal(0, .2, (5, 6)),
            'interaction': rng.normal(0, .2, 4)}


def test_candidate_affinity_matches_independent_4096_state_enumeration():
    params = fixture()
    q = unary(params)
    g = 1.5*expit(params['interaction'])/2
    logits = (np.log(q[0])[:, None, None] + np.log(q[1])[None, :, None]
              + np.log(q[2])[None, None, :])
    edge = (STATES*g) @ STATES.T
    logits += edge[:, :, None] + edge[:, None, :]
    weights = np.exp(logits-logits.max())
    p = weights.sum(axis=(1, 2))/weights.sum()
    expected = np.log(np.sum(np.sqrt(p[None, :]*q[[3, 4]]), axis=1))
    actual = ising_candidates(params, 0, [1, 2], np.array([3, 4]))
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual, ising_candidates(params, 0, [2, 1], np.array([3, 4])), atol=1e-12, rtol=0)
    expected_kernel = np.log(np.einsum('i,ij,kj->k', p, np.exp(edge), q[[3, 4]]))
    np.testing.assert_allclose(ising_kernel_candidates(params, 0, [1, 2], np.array([3, 4])), expected_kernel, atol=1e-12, rtol=0)


def test_affinity_identity_and_zero_coupling():
    params = fixture()
    params['interaction'] = np.full(4, -1000.)
    q = unary(params)
    assert abs(log_bhattacharyya(q[0], q[[0]])[0]) < 1e-12
    actual = ising_candidates(params, 0, [1, 2], np.array([3, 4]))
    expected = ising_candidates(params, 0, [2, 3], np.array([3, 4]), use_context=False)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(ising_kernel_candidates(params, 0, [1, 2], np.array([3, 4])), 0., atol=1e-12, rtol=0)
    context_only = ising_context_only_candidates(params, [1, 2], np.array([3, 4]))
    np.testing.assert_allclose(context_only, log_bhattacharyya(np.full(16, 1/16), q[[3, 4]]), atol=1e-12, rtol=0)
    with pytest.raises(ValueError):
        log_bhattacharyya(np.array([.1, .2]), np.array([[.5, .5]]))


def test_vector_modes_and_validation():
    vectors = {'shared': np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 0.]])}
    assert vector_candidates(vectors, 0, [1], np.array([2]))[0] == 1.
    assert vector_candidates(vectors, 0, [1], np.array([3]), mode='target')[0] == -1.
    assert abs(vector_candidates(vectors, 0, [1], np.array([3]), mode='context')[0]) < 1e-12
    assert vector_candidates(vectors, 0, [1], np.array([2]), similarity='dot')[0] == 2.
    with pytest.raises(ValueError):
        vector_candidates(vectors, 0, [], np.array([1]))
    with pytest.raises(ValueError):
        vector_candidates(vectors, 0, [3], np.array([1]))
