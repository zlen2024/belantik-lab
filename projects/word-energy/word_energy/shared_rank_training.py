"""Exact differentiable shared-star ranking primitives; no training or sampling."""
from word_energy.ising.exact_training import _imports
from word_energy.wikitext_four_spin import state_table, unary_probabilities


def validate_training_batch(windows, a_preferred, temperature, vocab_size):
    """Validate host arrays before passing a batch into JAX-compiled training."""
    import numpy as np
    w, labels, tau = np.asarray(windows), np.asarray(a_preferred), np.asarray(temperature)
    if w.ndim != 2 or w.shape[1] != 4 or not len(w) or w.dtype.kind not in 'iu':
        raise ValueError('windows must be a nonempty integer array with shape (N, 4)')
    if np.any(w < 0) or np.any(w >= vocab_size) or np.any(w[:, 2] == w[:, 3]):
        raise ValueError('invalid token ID or identical candidate pair')
    if labels.dtype.kind != 'b' or labels.shape != (len(w),):
        raise ValueError('labels must be boolean with shape (N,)')
    if tau.ndim != 0 or tau.dtype.kind not in 'iuf' or not np.isfinite(tau) or tau <= 0:
        raise ValueError('temperature must be a finite positive scalar')


def shared_pair_scores(params, windows):
    """Each row is target, one context, candidate A, candidate B.

    Both scores come from one target-centered Ising star. Per-spoke coupling
    is g/3; equal token IDs still denote separate spin occurrences.
    """
    jax, jnp, _ = _imports()
    windows = jnp.asarray(windows, dtype=jnp.int32)
    states = state_table(jnp)
    q = unary_probabilities(params)
    g = 1.5*jax.nn.sigmoid(params['interaction'])/3
    kernel = jnp.exp((states*g) @ states.T)
    messages = q @ kernel.T
    corr_messages = q @ (kernel*((states @ states.T)/4)).T
    center, context, a, b = (windows[:, i] for i in range(4))
    base = q[center]*messages[context]*messages[a]*messages[b]
    norm = jnp.sum(base, axis=1)
    def one(candidate):
        return jnp.sum(base*corr_messages[candidate]/messages[candidate], axis=1)/norm
    return jnp.stack((one(a), one(b)), axis=1)


def paired_rank_loss(params, windows, a_preferred, temperature):
    """Logistic ranking loss; caller must fix a finite positive temperature."""
    if isinstance(temperature, (int, float)) and not 0 < temperature < float('inf'):
        raise ValueError('temperature must be finite and positive')
    jax, jnp, _ = _imports()
    scores = shared_pair_scores(params, windows)
    sign = jnp.where(jnp.asarray(a_preferred, dtype=bool), 1., -1.)
    return jnp.mean(jax.nn.softplus(-sign*(scores[:, 0]-scores[:, 1])/temperature))
