"""Independent numerical and invariance checks for contextual readouts."""
import numpy as np
import pytest
from scipy.special import expit

from word_energy.contextual_substitution import (ising_active_correlations, ising_active_uniform_context_control, ising_candidates,
    ising_context_only_candidates, ising_kernel_candidates, ising_shared_correlations, log_bhattacharyya, vector_candidates)
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


def test_active_correlation_matches_joint_enumeration_and_zero_coupling():
    params = fixture()
    q = unary(params)
    g = 1.5*expit(params['interaction'])/2
    edge = (STATES*g) @ STATES.T
    logits = (np.log(q[0])[:, None, None] + np.log(q[1])[None, :, None]
              + np.log(q[3])[None, None, :] + edge[:, :, None] + edge[:, None, :])
    weights = np.exp(logits-logits.max())
    weights /= weights.sum()
    pair_correlation = (STATES @ STATES.T)/4
    expected = np.sum(weights*pair_correlation[:, None, :])
    actual = ising_active_correlations(params, 0, [1], np.array([3]))
    np.testing.assert_allclose(actual, [expected], atol=1e-12, rtol=0)
    params['interaction'] = np.full(4, -1000.)
    target_mean, candidate_mean = q[0] @ STATES, q[3] @ STATES
    zero_coupling = np.dot(target_mean, candidate_mean)/4
    np.testing.assert_allclose(ising_active_correlations(params, 0, [1], np.array([3])),
                               [zero_coupling], atol=1e-12, rtol=0)


def test_active_correlation_handles_repeated_word_occurrences():
    params = fixture()
    q = unary(params)
    g = 1.5*expit(params['interaction'])/3
    edge = (STATES*g) @ STATES.T
    # Target 0, two separate occurrences of context 1, and a separate
    # candidate occurrence of word 0 share parameters, not spin states.
    logits = (np.log(q[0])[:, None, None, None] + np.log(q[1])[None, :, None, None]
              + np.log(q[1])[None, None, :, None] + np.log(q[0])[None, None, None, :]
              + edge[:, :, None, None] + edge[:, None, :, None] + edge[:, None, None, :])
    weights = np.exp(logits-logits.max())
    weights /= weights.sum()
    expected = np.sum(weights*((STATES @ STATES.T)/4)[:, None, None, :])
    np.testing.assert_allclose(ising_active_correlations(params, 0, [1, 1], np.array([0])),
                               [expected], atol=1e-12, rtol=0)


def test_active_correlation_uniform_unary_known_answer():
    params = {'biases': np.zeros((3, 4)), 'internal': np.zeros((3, 6)),
              'interaction': np.zeros(4)}
    # Each edge has coupling 0.75/2 and uniform independent unary spins.
    np.testing.assert_allclose(ising_active_correlations(params, 0, [1], np.array([2])),
                               [np.tanh(.75/2)], atol=1e-12, rtol=0)


def test_uniform_context_control_matches_pair_at_original_spoke_strength():
    params = fixture()
    q = unary(params)
    m = 3
    coupling = 1.5*expit(params['interaction'])/(m+1)
    edge = (STATES*coupling) @ STATES.T
    weights = q[0][:, None]*q[3][None, :]*np.exp(edge)
    weights /= weights.sum()
    expected = np.sum(weights*((STATES @ STATES.T)/4))
    actual = ising_active_uniform_context_control(params, 0, m, np.array([3]))
    np.testing.assert_allclose(actual, [expected], atol=1e-12, rtol=0)
    with pytest.raises(ValueError):
        ising_active_uniform_context_control(params, 0, 0, np.array([3]))


def test_shared_correlations_match_four_token_joint_and_permute_candidates():
    params = fixture()
    q = unary(params)
    coupling = 1.5*expit(params['interaction'])/3
    edge = (STATES*coupling) @ STATES.T
    logits = (np.log(q[0])[:, None, None, None] + np.log(q[1])[None, :, None, None]
              + np.log(q[3])[None, None, :, None] + np.log(q[4])[None, None, None, :]
              + edge[:, :, None, None] + edge[:, None, :, None] + edge[:, None, None, :])
    weights = np.exp(logits-logits.max())
    weights /= weights.sum()
    product = (STATES @ STATES.T)/4
    expected = np.array([np.sum(weights*product[:, None, :, None]),
                         np.sum(weights*product[:, None, None, :])])
    actual = ising_shared_correlations(params, 0, [1], np.array([3, 4]))
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(ising_shared_correlations(params, 0, [1], np.array([4, 3])), expected[::-1], atol=1e-12, rtol=0)


def test_shared_correlations_repeated_ids_and_zero_coupling():
    params = fixture()
    one_candidate = ising_shared_correlations(params, 0, [1, 1], np.array([3]))
    np.testing.assert_allclose(one_candidate, ising_active_correlations(params, 0, [1, 1], np.array([3])), atol=1e-12, rtol=0)
    duplicate = ising_shared_correlations(params, 0, [1], np.array([0, 0]))
    np.testing.assert_allclose(duplicate[0], duplicate[1], atol=1e-12, rtol=0)
    params['interaction'] = np.full(4, -1000.)
    q = unary(params)
    means = q @ STATES
    expected = means[[0, 3]] @ means[0]/4
    np.testing.assert_allclose(ising_shared_correlations(params, 0, [1], np.array([0, 3])), expected, atol=1e-12, rtol=0)
    with pytest.raises(ValueError):
        ising_shared_correlations(params, 0, [1], np.array([], dtype=int))
    with pytest.raises(ValueError):
        ising_shared_correlations(params, 0, [1.5], np.array([3]))
    with pytest.raises(ValueError):
        ising_shared_correlations(params, 0, [1], np.array([3.5]))
    with pytest.raises(ValueError):
        ising_shared_correlations(params, True, [1], np.array([3]))


def test_shared_correlations_can_represent_crossed_rank_reversals():
    # Hand-set witness of representational capacity, not learned semantics.
    coupling = 3*np.arctanh(1/3)
    params = {'biases': np.zeros((6, 4)), 'internal': np.zeros((6, 6)),
              'interaction': np.full(4, np.log(coupling/(1.5-coupling)))}
    for token, coordinate, mean in ((0, 0, -.8), (1, 0, .8), (2, 1, -.8), (3, 1, .8)):
        params['biases'][token, coordinate] = np.arctanh(np.arctanh(mean)/2.5)
    params['internal'][4, 0] = np.arctanh(np.arctanh(.8)/2)
    params['internal'][5, 0] = np.arctanh(np.arctanh(-.8)/2)
    scores = [ising_shared_correlations(params, target, [context], np.array([4, 5]))
              for target, context in ((0, 2), (0, 3), (1, 2), (1, 3))]
    differences = np.array([score[0]-score[1] for score in scores])
    np.testing.assert_allclose(differences, np.array([1, -1, -1, 1])*512/10045, atol=1e-12, rtol=0)
