import numpy as np

from word_energy.ising.exact_training import _free_energy_factory


def test_exact_free_energy_autodiff_matches_central_difference() -> None:
    jax, jnp, free_energy, _, _ = _free_energy_factory()
    raw = {
        "biases": jnp.array([[0.1, -0.2], [0.3, 0.05], [-0.1, 0.2]], dtype=jnp.float64),
        "internal": jnp.array([0.2, -0.1, 0.05], dtype=jnp.float64),
        "interaction": jnp.array([0.2, -0.3], dtype=jnp.float64),
    }
    # Token 0 repeats, so this covers repeated-occurrence accounting too.
    tokens = jnp.array([0, 0, 2], dtype=jnp.int32)
    step = 1e-5
    for name, index in (("biases", (0, 0)), ("internal", 0), ("interaction", 0)):
        analytic = jax.grad(lambda value: free_energy({**raw, name: value}, tokens))(raw[name])[index]
        plus = raw[name].at[index].add(step)
        minus = raw[name].at[index].add(-step)
        numeric = (free_energy({**raw, name: plus}, tokens) - free_energy({**raw, name: minus}, tokens)) / (2 * step)
        assert np.isclose(float(analytic), float(numeric), atol=1e-6, rtol=1e-4)


def test_normalised_exact_score_is_zero_when_lambda_is_zero() -> None:
    _, jnp, free_energy, _, _ = _free_energy_factory()
    raw = {
        "biases": jnp.array([[0.1, -0.2], [0.3, 0.05]], dtype=jnp.float64),
        "internal": jnp.array([0.2, -0.1], dtype=jnp.float64),
        "interaction": jnp.array([0.2, -0.3], dtype=jnp.float64),
    }
    assert abs(float(free_energy(raw, jnp.array([0, 1], dtype=jnp.int32), 0.0))) < 1e-12
    assert abs(float(free_energy(raw, jnp.array([0, 1, 0], dtype=jnp.int32), 0.0))) < 1e-12
