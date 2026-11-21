"""Exact differentiable context-effect scores for a proposed training objective.

These functions use analytic state sums, not a sampler or a training loop.
Each window is target, one context and one candidate. The candidate is active
in both stars; only the context unary is replaced by a uniform distribution.
"""
from __future__ import annotations

from word_energy.ising.exact_training import _imports
from word_energy.wikitext_four_spin import state_table, unary_probabilities


def ising_context_contrast(params, windows, *, uniform_target=False):
    """Return active R, uniform-context R and their difference per window."""
    jax, jnp, _ = _imports()
    windows = jnp.asarray(windows, dtype=jnp.int32)
    states = state_table(jnp)
    q = unary_probabilities(params)
    g = 1.5*jax.nn.sigmoid(params['interaction'])/2
    kernel = jnp.exp((states*g) @ states.T)
    product = (states @ states.T)/4
    messages = q @ kernel.T
    correlation_messages = q @ (kernel*product).T
    target, context, candidate = windows[:, 0], windows[:, 1], windows[:, 2]
    base = jnp.full_like(q[target], 1/states.shape[0]) if uniform_target else q[target]
    weighted = base*messages[context]

    def correlation(target_weights):
        numerator = jnp.sum(target_weights*correlation_messages[candidate], axis=1)
        denominator = jnp.sum(target_weights*messages[candidate], axis=1)
        return numerator/denominator

    active = correlation(weighted)
    uniform = correlation(base)
    return active, uniform, active-uniform


def ising_target_context_interaction(params, windows):
    """Subtract context effect with a uniform target occurrence."""
    return ising_context_contrast(params, windows)[2] - ising_context_contrast(params, windows, uniform_target=True)[2]


def vector_context_contrast(params, windows, *, zero_target=False):
    """Target-context cosine minus target-only cosine with smooth norm floor."""
    _, jnp, _ = _imports()
    windows = jnp.asarray(windows, dtype=jnp.int32)
    vectors = params['shared']
    target, context, candidate = vectors[windows[:, 0]], vectors[windows[:, 1]], vectors[windows[:, 2]]
    if zero_target:
        target = jnp.zeros_like(target)

    def cosine(left, right):
        numerator = jnp.sum(left*right, axis=1)
        denominator = jnp.sqrt(jnp.sum(left*left, axis=1)+1e-12)*jnp.sqrt(jnp.sum(right*right, axis=1)+1e-12)
        return numerator/denominator

    active = cosine(target+context, candidate)
    uniform = cosine(target, candidate)
    return active, uniform, active-uniform


def vector_target_context_interaction(params, windows):
    """Subtract context effect after removing the target vector occurrence."""
    return vector_context_contrast(params, windows)[2] - vector_context_contrast(params, windows, zero_target=True)[2]


def ising_target_foil_gap(params, windows, foil_ids):
    """Difference in context effect for observed versus foil target IDs."""
    _, jnp, _ = _imports()
    windows = jnp.asarray(windows, dtype=jnp.int32)
    foil = windows.at[:, 0].set(jnp.asarray(foil_ids, dtype=jnp.int32))
    return ising_context_contrast(params, windows)[2] - ising_context_contrast(params, foil)[2]


def vector_target_foil_gap(params, windows, foil_ids):
    """Tied-vector context-effect gap under the same target replacement."""
    _, jnp, _ = _imports()
    windows = jnp.asarray(windows, dtype=jnp.int32)
    foil = windows.at[:, 0].set(jnp.asarray(foil_ids, dtype=jnp.int32))
    return vector_context_contrast(params, windows)[2] - vector_context_contrast(params, foil)[2]
