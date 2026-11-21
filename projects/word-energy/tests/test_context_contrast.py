"""Numerical parity and gradient checks for the proposed context-effect score."""
import numpy as np

from word_energy.context_contrast import (ising_context_contrast, ising_target_context_interaction,
    ising_target_foil_gap, vector_context_contrast, vector_target_context_interaction, vector_target_foil_gap)
from word_energy.contextual_substitution import ising_active_correlations, ising_active_uniform_context_control
from word_energy.ising.exact_training import _imports


def fixture():
    rng = np.random.default_rng(913)
    return {'biases': rng.normal(0, .2, (5, 4)), 'internal': rng.normal(0, .2, (5, 6)),
            'interaction': rng.normal(0, .2, 4)}


def test_exact_score_parity_with_production_readout_and_repeated_ids():
    params = fixture()
    windows = np.array([[0, 1, 2], [0, 0, 0], [3, 2, 3]], dtype=np.int32)
    active, uniform, contrast = (np.asarray(x) for x in ising_context_contrast(params, windows))
    expected = np.array([ising_active_correlations(params, int(t), [int(c)], np.array([k]))[0] for t,c,k in windows])
    no_context = np.array([ising_active_uniform_context_control(params, int(t), 1, np.array([k]))[0] for t,c,k in windows])
    np.testing.assert_allclose(active, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(uniform, no_context, atol=1e-12, rtol=0)
    np.testing.assert_allclose(contrast, expected-no_context, atol=1e-12, rtol=0)


def test_gradient_finite_difference_all_parameter_types():
    jax, jnp, _ = _imports()
    params = {k: jnp.asarray(v) for k,v in fixture().items()}
    windows = jnp.asarray([[0, 1, 2], [0, 0, 0], [3, 2, 3]], dtype=jnp.int32)
    for score in (lambda p: ising_context_contrast(p, windows)[2],
                  lambda p: ising_target_context_interaction(p, windows)):
        def objective(p):
            return jnp.sum(score(p))
        gradients = jax.grad(objective)(params)
        for name, index in (('biases', (0, 0)), ('internal', (0, 0)), ('interaction', (0,))):
            plus, minus = dict(params), dict(params)
            plus[name] = params[name].at[index].add(1e-5)
            minus[name] = params[name].at[index].add(-1e-5)
            numeric = (float(objective(plus))-float(objective(minus)))/(2e-5)
            np.testing.assert_allclose(float(gradients[name][index]), numeric, atol=1e-7, rtol=1e-4)


def test_vector_contrast_is_finite_at_zero_and_cancels_without_context():
    params = {'shared': np.array([[1., 0.], [0., 0.], [0., 1.]])}
    active, baseline, contrast = (np.asarray(x) for x in vector_context_contrast(params, np.array([[0, 1, 2], [1, 1, 1]])))
    assert np.isfinite(active).all()
    np.testing.assert_allclose(active, baseline, atol=1e-12, rtol=0)
    np.testing.assert_allclose(contrast, 0., atol=1e-12, rtol=0)


def test_target_removal_cancels_context_candidate_shortcut():
    params = fixture()
    params['biases'][0] = 0
    params['internal'][0] = 0
    windows = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int32)
    np.testing.assert_allclose(ising_target_context_interaction(params, windows), 0., atol=1e-12, rtol=0)
    vectors = {'shared': np.array([[0., 0.], [1., 0.], [0., 1.], [-1., 0.], [1., 1.]])}
    np.testing.assert_allclose(vector_target_context_interaction(vectors, windows), 0., atol=1e-12, rtol=0)


def test_target_removal_does_not_force_target_identity_or_raw_r_ranking():
    # Two target IDs deliberately have identical, nonuniform unary laws.
    # Only coordinate 0 has fields. The other three spin contributions to
    # R cancel from the contrasts, so one-spin values are divided by four.
    params = {'biases': np.zeros((6, 4)), 'internal': np.zeros((6, 6)),
              'interaction': np.full(4, np.log((2*np.arctanh(.5))/(1.5-2*np.arctanh(.5))))}
    for token, mean in ((0, .5), (4, .5), (1, .8), (2, -.8), (3, .8)):
        params['biases'][token, 0] = np.arctanh(np.arctanh(mean)/2.5)
    windows = np.array([[0, 1, 2], [0, 1, 3], [4, 1, 2], [4, 1, 3]], dtype=np.int32)
    active, _, _ = (np.asarray(x) for x in ising_context_contrast(params, windows))
    interaction = np.asarray(ising_target_context_interaction(params, windows))
    explicit_foil_gap = np.asarray(ising_target_foil_gap(params, windows, np.full(4, 5)))
    np.testing.assert_allclose(explicit_foil_gap, interaction, atol=1e-12, rtol=0)
    np.testing.assert_allclose(interaction, np.array([1/56, -167/1508, 1/56, -167/1508])/4, atol=1e-12, rtol=0)
    assert interaction[0] > interaction[1]
    assert active[0] < active[1]
    np.testing.assert_allclose(interaction[:2], interaction[2:], atol=1e-12, rtol=0)
    # Both IDs appear once as observed target and once as foil. Even under
    # that balancing, the foil gaps can be positive while raw R ranks the
    # two candidates in the opposite direction for the nonuniform target.
    balanced = np.array([[0, 1, 2], [5, 1, 3]], dtype=np.int32)
    balanced_gap = np.asarray(ising_target_foil_gap(params, balanced, np.array([5, 0])))
    np.testing.assert_allclose(balanced_gap, [1/224, 167/6032], atol=1e-12, rtol=0)
    assert np.all(balanced_gap > 0)


def test_target_foil_gap_rejects_identical_target_representations():
    params = fixture()
    params['biases'][4] = params['biases'][0]
    params['internal'][4] = params['internal'][0]
    windows = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int32)
    np.testing.assert_allclose(ising_target_foil_gap(params, windows, np.array([0, 4])), 0., atol=1e-12, rtol=0)
    vectors = {'shared': np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 0.], [1., 0.]])}
    np.testing.assert_allclose(vector_target_foil_gap(vectors, windows, np.array([0, 4])), 0., atol=1e-12, rtol=0)


def test_target_foil_gap_gradient_matches_finite_difference():
    jax, jnp, _ = _imports()
    params = {k: jnp.asarray(v) for k, v in fixture().items()}
    windows = jnp.asarray([[0, 1, 2], [2, 2, 3]], dtype=jnp.int32)
    foil_ids = jnp.asarray([4, 0], dtype=jnp.int32)
    def objective(p):
        return jnp.sum(jax.nn.softplus(.1-ising_target_foil_gap(p, windows, foil_ids)))
    gradient = jax.grad(objective)(params)
    for name, index in (('biases', (0, 0)), ('internal', (0, 0)), ('interaction', (0,))):
        plus, minus = dict(params), dict(params)
        plus[name] = params[name].at[index].add(1e-5)
        minus[name] = params[name].at[index].add(-1e-5)
        numeric = (float(objective(plus))-float(objective(minus)))/(2e-5)
        np.testing.assert_allclose(float(gradient[name][index]), numeric, atol=1e-7, rtol=1e-4)
