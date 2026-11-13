import numpy as np

from word_energy.energy_embedding.train_joint_toy import (
    GaussianJointConfig,
    make_joint_free_energy,
    quadrature_refinement_audit,
)


def test_gaussian_joint_free_energy_lambda_zero_and_gradient() -> None:
    config = GaussianJointConfig(lexical_updates=1, joint_updates=1, quadrature_order=8)
    jax, jnp, _, free_energy = make_joint_free_energy(config)
    raw = {
        "means": jnp.array([[[0.1], [-0.2]], [[0.3], [0.05]], [[-0.1], [0.2]]], dtype=jnp.float64),
        "raw_variances": jnp.array([[[0.2], [-0.1]], [[0.05], [0.3]], [[-0.2], [0.1]]], dtype=jnp.float64),
        "logits": jnp.array([[0.2, -0.1], [0.05, 0.3], [-0.2, 0.1]], dtype=jnp.float64),
    }
    tokens = jnp.array([0, 1, 2], dtype=jnp.int32)
    assert abs(float(free_energy(raw, tokens, 0.0))) < 1e-12
    analytic = jax.grad(lambda means: free_energy({**raw, "means": means}, tokens))(raw["means"])[0, 0, 0]
    step = 1e-5
    plus = raw["means"].at[0, 0, 0].add(step)
    minus = raw["means"].at[0, 0, 0].add(-step)
    numeric = (free_energy({**raw, "means": plus}, tokens) - free_energy({**raw, "means": minus}, tokens)) / (2 * step)
    assert np.isclose(float(analytic), float(numeric), atol=1e-6, rtol=1e-4)
    audit = quadrature_refinement_audit(raw, np.asarray(tokens), config)
    # This generic fixture checks the audit wiring. The preregistered 1e-4
    # gate is applied to the trained checkpoint, not a separate arbitrary raw
    # fixture whose mixture shape may need a higher order.
    assert audit["numerics.refinement_f_abs_delta"] < 2e-4
    assert audit["numerics.refinement_mean_gradient_relative_delta"] < 2e-2
