"""Temporary training-only diversity penalty for a targeted token.

This Shannon entropy in nats is a mathematical diagnostic, not hardware heat.
The shared model energy and joint ranking loss are deliberately unchanged.
"""
import math


def schedule_weight(update: int) -> float:
    if 0 <= update < 50:
        return 1.0
    if 50 <= update < 100:
        return (100-update)/50
    return 0.0


def entropy_kl_to_uniform(raw, token_id):
    from word_energy.ising.exact_training import _imports
    jax, jnp, _ = _imports()
    # Reconstruct log-probabilities from the same Ising scores, avoiding log(0).
    states = jnp.asarray([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]])
    bias = 2.5*jnp.tanh(raw['biases'][token_id])
    internal = 2.0*jnp.tanh(raw['internal'][token_id])
    scores = states @ bias + internal*states[:, 0]*states[:, 1]
    log_q = jax.nn.log_softmax(scores)
    q = jnp.exp(log_q)
    return math.log(4.) + jnp.sum(q*log_q)
