import numpy as np

from word_energy.energy_embedding.train_lexical import (
    GaussianLexicalConfig,
    _compatibility_energy,
    _constrained,
    _imports,
)


def test_gaussian_analytic_overlap_gradient_matches_central_difference() -> None:
    jax, jnp, _ = _imports()
    config = GaussianLexicalConfig()
    raw = {
        "means": jnp.array([[[0.1], [-0.2]], [[0.3], [0.05]]], dtype=jnp.float64),
        "raw_variances": jnp.array([[[0.2], [-0.1]], [[0.05], [0.3]]], dtype=jnp.float64),
        "logits": jnp.array([[0.2, -0.1], [0.05, 0.3]], dtype=jnp.float64),
    }
    analytic = jax.grad(lambda means: _compatibility_energy({**raw, "means": means}, 0, 1, config, jnp))(raw["means"])[0, 0, 0]
    step = 1e-5
    plus = raw["means"].at[0, 0, 0].add(step)
    minus = raw["means"].at[0, 0, 0].add(-step)
    numeric = (
        _compatibility_energy({**raw, "means": plus}, 0, 1, config, jnp)
        - _compatibility_energy({**raw, "means": minus}, 0, 1, config, jnp)
    ) / (2 * step)
    assert np.isclose(float(analytic), float(numeric), atol=1e-6, rtol=1e-4)


def test_gaussian_constraints_bound_means_and_variance() -> None:
    _, jnp, _ = _imports()
    config = GaussianLexicalConfig()
    raw = {
        "means": jnp.array([[[10.0], [-10.0]]], dtype=jnp.float64),
        "raw_variances": jnp.array([[[-100.0], [100.0]]], dtype=jnp.float64),
        "logits": jnp.zeros((1, 2), dtype=jnp.float64),
    }
    constrained = _constrained(raw, config, jnp)
    assert np.all(np.linalg.norm(np.asarray(constrained["means"]), axis=-1) <= 3.0 + 1e-12)
    assert np.all(np.asarray(constrained["variances"]) >= 0.16)
    assert np.all(np.asarray(constrained["variances"]) <= 1.44)


def test_gaussian_constraints_support_preregistered_capacity_shapes() -> None:
    _, jnp, _ = _imports()
    for components in (1, 4):
        config = GaussianLexicalConfig(components=components)
        raw = {
            "means": jnp.zeros((2, components, 1), dtype=jnp.float64),
            "raw_variances": jnp.zeros((2, components, 1), dtype=jnp.float64),
            "logits": jnp.zeros((2, components), dtype=jnp.float64),
        }
        constrained = _constrained(raw, config, jnp)
        assert constrained["means"].shape == raw["means"].shape
