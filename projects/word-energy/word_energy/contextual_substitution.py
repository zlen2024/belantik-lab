"""Candidate compatibility readouts for frozen contextual word representations.

Affinity and kernel readouts compare the target's active-window posterior
with an external candidate. The separate active-correlation readout places
each candidate in its own window and computes its joint spin correlation.
The shared-correlation readout instead places all candidates in one star;
its scores depend on the entire candidate set and its coupling normalization.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp

from word_energy.wic_context_probe import cosine
from word_energy.wikitext_four_spin_wic import STATES, posterior, unary


def log_bhattacharyya(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Log affinity of one normalized positive distribution to each candidate."""
    reference = np.asarray(reference, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    if reference.ndim != 1 or candidates.ndim != 2 or candidates.shape[1] != reference.size:
        raise ValueError('expected one reference and candidate probability rows')
    if not np.isfinite(reference).all() or not np.isfinite(candidates).all():
        raise ValueError('probabilities must be finite')
    if np.any(reference <= 0) or np.any(candidates <= 0):
        raise ValueError('probabilities must be strictly positive')
    if not np.isclose(reference.sum(), 1., rtol=0, atol=1e-12) or not np.allclose(candidates.sum(axis=1), 1., rtol=0, atol=1e-12):
        raise ValueError('probabilities must be normalized')
    scores = logsumexp(.5*(np.log(reference)[None, :] + np.log(candidates)), axis=1)
    if np.any(scores > 1e-12):
        raise FloatingPointError('affinity exceeds one')
    return np.minimum(scores, 0.)


def ising_candidates(params: dict[str, np.ndarray], target: int, contexts: list[int],
                     candidates: np.ndarray, *, use_context: bool = True) -> np.ndarray:
    """Compare target state distributions to external candidate word landscapes."""
    q = unary(params)
    ids = np.asarray(candidates, dtype=np.int64)
    if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(q)) or not 0 <= target < len(q):
        raise ValueError('invalid target or candidate IDs')
    if not contexts or any(c < 0 or c >= len(q) for c in contexts):
        raise ValueError('contexts must be nonempty valid token IDs')
    ref = posterior(q, 1.5*expit(params['interaction']), target, contexts) if use_context else q[target]
    return log_bhattacharyya(ref, q[ids])


def ising_kernel_candidates(params: dict[str, np.ndarray], target: int, contexts: list[int],
                            candidates: np.ndarray, *, use_context: bool = True) -> np.ndarray:
    """Host-side pair compatibility between target posterior and external candidates.

    Candidate couplings use half the trained coupling, matching one spoke in
    the three-token training graph. The target posterior is computed first,
    so candidates do not influence that active-window equilibrium.
    """
    q = unary(params)
    ids = np.asarray(candidates, dtype=np.int64)
    if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(q)) or not 0 <= target < len(q):
        raise ValueError('invalid target or candidate IDs')
    if not contexts or any(c < 0 or c >= len(q) for c in contexts):
        raise ValueError('contexts must be nonempty valid token IDs')
    coupling = 1.5*expit(params['interaction'])
    ref = posterior(q, coupling, target, contexts) if use_context else q[target]
    pair_logits = (STATES*(coupling/2)) @ STATES.T
    return logsumexp(np.log(ref)[None, :, None] + pair_logits[None, :, :]
                     + np.log(q[ids])[:, None, :], axis=(1, 2))


def ising_context_only_candidates(params: dict[str, np.ndarray], contexts: list[int],
                                  candidates: np.ndarray) -> np.ndarray:
    """Compare candidates to a contextual posterior with a uniform center prior."""
    q = unary(params)
    ids = np.asarray(candidates, dtype=np.int64)
    if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(q)):
        raise ValueError('invalid candidate IDs')
    if not contexts or any(c < 0 or c >= len(q) for c in contexts):
        raise ValueError('contexts must be nonempty valid token IDs')
    extended = np.vstack((q, np.full(q.shape[1], 1/q.shape[1])))
    reference = posterior(extended, 1.5*expit(params['interaction']), len(q), contexts)
    return log_bhattacharyya(reference, q[ids])


def ising_active_correlations(params: dict[str, np.ndarray], target: int, contexts: list[int],
                              candidates: np.ndarray) -> np.ndarray:
    """Exact target-candidate spin correlations within each candidate's star.

    Each candidate is an active leaf in its own target-centered window along
    with the provided contexts. Other candidate words are not active in that
    window. All spokes use the production coupling divided by their count.
    Adding the candidate changes both the equilibrium and each context spoke
    from g/m to g/(m+1), where m is the original number of context tokens.
    """
    q = unary(params)
    ids = np.asarray(candidates, dtype=np.int64)
    if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(q)) or not 0 <= target < len(q):
        raise ValueError('invalid target or candidate IDs')
    if not contexts or any(c < 0 or c >= len(q) for c in contexts):
        raise ValueError('contexts must be nonempty valid token IDs')
    coupling = 1.5*expit(params['interaction'])
    edge = (STATES * (coupling/(len(contexts)+1))) @ STATES.T
    products = (STATES @ STATES.T)/STATES.shape[1]
    scores = []
    for candidate in ids:
        reference = posterior(q, coupling, target, [*contexts, int(candidate)])
        log_weights = edge + np.log(q[candidate])[None, :]
        conditional = np.exp(log_weights-logsumexp(log_weights, axis=1, keepdims=True))
        scores.append(float(np.sum(reference[:, None]*conditional*products)))
    return np.asarray(scores)


def ising_active_uniform_context_control(params: dict[str, np.ndarray], target: int,
                                         context_count: int, candidates: np.ndarray) -> np.ndarray:
    """Remove context identity while retaining active-star spoke normalization."""
    if context_count < 1:
        raise ValueError('context count must be positive')
    expanded = dict(params)
    expanded['biases'] = np.vstack((params['biases'], np.zeros((1, 4))))
    expanded['internal'] = np.vstack((params['internal'], np.zeros((1, 6))))
    return ising_active_correlations(expanded, target, [len(params['biases'])]*context_count, candidates)


def ising_shared_correlations(params: dict[str, np.ndarray], target: int, contexts: list[int],
                              candidates: np.ndarray) -> np.ndarray:
    """Exact target-candidate correlations in one shared target-centered star.

    Every context and candidate is a distinct active spin occurrence, even
    when word IDs repeat. Couplings are g divided by the total leaf count.
    Adding or removing any leaf changes the joint system and all spoke scales.
    """
    q = unary(params)
    ids = np.asarray(candidates)
    context_ids = np.asarray(contexts)
    if not isinstance(target, (int, np.integer)) or isinstance(target, (bool, np.bool_)):
        raise ValueError('target must be an integer token ID')
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids) or np.any(ids < 0) or np.any(ids >= len(q)) or not 0 <= target < len(q):
        raise ValueError('invalid target or candidate IDs')
    if context_ids.ndim != 1 or (len(context_ids) and context_ids.dtype.kind not in 'iu'):
        raise ValueError('contexts must contain integer token IDs')
    if np.any(context_ids < 0) or np.any(context_ids >= len(q)):
        raise ValueError('invalid context token IDs')
    ids = ids.astype(np.int64, copy=False)
    leaves = np.concatenate((context_ids.astype(np.int64, copy=False), ids))
    coupling = 1.5*expit(params['interaction'])/len(leaves)
    edge = (STATES*coupling) @ STATES.T
    log_messages = logsumexp(edge[None, :, :] + np.log(q[leaves])[:, None, :], axis=2)
    log_center = np.log(q[target]) + np.sum(log_messages, axis=0)
    center = np.exp(log_center-logsumexp(log_center))
    log_candidate = edge[None, :, :] + np.log(q[ids])[:, None, :]
    conditional = np.exp(log_candidate-logsumexp(log_candidate, axis=2, keepdims=True))
    pair_products = (STATES @ STATES.T)/STATES.shape[1]
    return np.sum(center[None, :, None]*conditional*pair_products[None, :, :], axis=(1, 2))


def vector_candidates(params: dict[str, np.ndarray], target: int, contexts: list[int],
                      candidates: np.ndarray, *, mode: str = 'target_context',
                      similarity: str = 'cosine') -> np.ndarray:
    """Cosine or dot-product controls on the same tied word table and candidates."""
    vectors = np.asarray(params['shared'], dtype=np.float64)
    ids = np.asarray(candidates, dtype=np.int64)
    if vectors.ndim != 2 or ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(vectors)) or not 0 <= target < len(vectors):
        raise ValueError('invalid vector table, target or candidate IDs')
    if not contexts or any(c < 0 or c >= len(vectors) for c in contexts):
        raise ValueError('contexts must be nonempty valid token IDs')
    if mode not in ('target_context', 'target', 'context'):
        raise ValueError('unknown vector readout mode')
    if similarity not in ('cosine', 'dot'):
        raise ValueError('unknown vector similarity')
    reference = vectors[target] if mode == 'target' else np.mean(vectors[contexts], axis=0)
    if mode == 'target_context':
        reference = reference + vectors[target]
    if similarity == 'dot':
        return vectors[ids] @ reference
    return np.asarray([cosine(reference, vectors[c]) for c in ids])
