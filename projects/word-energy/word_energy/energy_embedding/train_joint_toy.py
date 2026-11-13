"""Differentiable Gauss--Hermite joint-training precheck for Gaussian Word Energy.

This module trains the D=1/M=2 toy using the *normalised* joint free energy
``F3=-log E_product_unary[exp(interaction)]``.  The quadrature sum is exact
for its fixed Gauss--Hermite approximation and differentiable through mixture
parameters; no MCMC or discrete sampler gradient is used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from word_energy.energy_embedding.train_lexical import (
    GaussianLexicalConfig,
    _batch_loss,
    _constrained,
    _copy,
    _imports,
    jax_log_softmax,
    jax_logsumexp,
    train_lexical,
)
from word_energy.ising.exact_training import prepare_data


@dataclass(frozen=True)
class GaussianJointConfig:
    seed: int = 17
    lexical_updates: int = 400
    joint_updates: int = 100
    lexical_batch: int = 64
    joint_batch: int = 4
    lexical_lr: float = 0.01
    joint_lr: float = 0.003
    clip_norm: float = 5.0
    validation_every: int = 20
    quadrature_order: int = 8
    coupling: float = 1.0
    length_scale: float = 1.0


def _quadrature_constants(order: int, components: int, window: int = 3):
    if order < 2:
        raise ValueError("quadrature order must be at least two")
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return (
        np.asarray(list(product(range(components), repeat=window)), dtype=np.int32),
        np.asarray(list(product(nodes, repeat=window)), dtype=np.float64),
        np.prod(
            np.asarray(list(product(weights / np.sqrt(np.pi), repeat=window)), dtype=np.float64), axis=1
        ),
    )


def make_joint_free_energy(config: GaussianJointConfig):
    """Build F3 over 3-token windows; constants make its 4,096 terms explicit."""
    jax, jnp, _ = _imports()
    assignments_np, node_grid_np, quadrature_weights_np = _quadrature_constants(config.quadrature_order, 2)
    assignments = jnp.asarray(assignments_np)
    node_grid = jnp.asarray(node_grid_np)
    log_quadrature_weights = jnp.log(jnp.asarray(quadrature_weights_np))
    positions = jnp.broadcast_to(jnp.arange(3), assignments.shape)
    lexical_config = GaussianLexicalConfig(
        seed=config.seed,
        updates=config.lexical_updates,
        batch_size=config.lexical_batch,
        learning_rate=config.lexical_lr,
        clip_norm=config.clip_norm,
        validation_every=config.validation_every,
    )

    def free_energy(raw, tokens, coupling_scale: float = 1.0):
        parameters = _constrained(raw, lexical_config, jnp)
        token_means = parameters["means"][tokens]
        token_variances = parameters["variances"][tokens]
        means = token_means[positions, assignments]
        standard_deviations = jnp.sqrt(token_variances[positions, assignments])
        states = means[:, None, :, :] + jnp.sqrt(2.0) * standard_deviations[:, None, :, :] * node_grid[None, :, :, None]
        log_mixture_weights = jnp.sum(
            jnp.stack(
                [jax_log_softmax(parameters["logits"][tokens[position]], jnp)[assignments[:, position]] for position in range(3)]
            ),
            axis=0,
        )
        interaction = jnp.zeros((assignments.shape[0], node_grid.shape[0]), dtype=jnp.float64)
        for left in range(3):
            for right in range(left + 1, 3):
                delta = states[:, :, left, 0] - states[:, :, right, 0]
                interaction = interaction + jnp.exp(-(delta * delta) / (2.0 * config.length_scale**2))
        interaction = coupling_scale * config.coupling * interaction / 2.0
        return -jax_logsumexp(log_mixture_weights[:, None] + log_quadrature_weights[None, :] + interaction, jnp)

    return jax, jnp, lexical_config, free_energy


def quadrature_refinement_audit(raw: dict[str, np.ndarray], tokens: np.ndarray, config: GaussianJointConfig) -> dict[str, float]:
    """Compare smooth F and its raw-parameter gradient at two GH resolutions.

    This does not validate discontinuous sense-region indicators. It is the
    numerical gate for the differentiable free-energy ranking objective only.
    """
    coarse_config = replace(config, quadrature_order=config.quadrature_order)
    refined_config = replace(config, quadrature_order=max(12, config.quadrature_order + 4))
    jax, jnp, _, coarse_energy = make_joint_free_energy(coarse_config)
    _, _, _, refined_energy = make_joint_free_energy(refined_config)
    parameters = {name: jnp.asarray(value) for name, value in raw.items()}
    token_ids = jnp.asarray(tokens, dtype=jnp.int32)
    coarse_f = float(coarse_energy(parameters, token_ids))
    refined_f = float(refined_energy(parameters, token_ids))
    coarse_gradient = jax.grad(lambda value: coarse_energy({**parameters, "means": value}, token_ids))(parameters["means"])
    refined_gradient = jax.grad(lambda value: refined_energy({**parameters, "means": value}, token_ids))(parameters["means"])
    coarse_norm = float(jnp.linalg.norm(coarse_gradient))
    gradient_delta = float(jnp.linalg.norm(coarse_gradient - refined_gradient))
    return {
        "numerics.refinement_coarse_order": float(coarse_config.quadrature_order),
        "numerics.refinement_f_abs_delta": abs(coarse_f - refined_f),
        "numerics.refinement_mean_gradient_relative_delta": gradient_delta / max(coarse_norm, 1e-6),
    }


def train_joint_precheck(config: GaussianJointConfig = GaussianJointConfig()):
    """Train lexical then joint rankings; select by validation joint loss only.

    Sense-region integration is deliberately not used as a selection metric yet:
    its discontinuous indicator needs its own refinement audit.
    """
    jax, jnp, optax = _imports()
    _, _, lexical_config, free_energy = make_joint_free_energy(config)
    data = prepare_data()
    lexical_raw, lexical_metrics = train_lexical(lexical_config)
    raw = {name: jnp.asarray(value) for name, value in lexical_raw.items()}
    optimizer = optax.chain(optax.clip_by_global_norm(config.clip_norm), optax.adam(config.joint_lr))
    state = optimizer.init(raw)
    windows = jnp.asarray(data["train_windows"])
    corruptions = jnp.asarray(data["train_corruptions"])
    pairs = jnp.asarray(data["train_pairs"])
    negatives = jnp.asarray(data["train_negatives"])
    validation_windows = jnp.asarray(data["validation_windows"])
    validation_corruptions = jnp.asarray(data["validation_corruptions"])
    validation_pairs = jnp.asarray(data["validation_pairs"])
    validation_negatives = jnp.asarray(data["validation_negatives"])

    def joint_loss(parameters, window_batch, corrupt_batch, pair_batch, negative_batch):
        positive = jax.vmap(lambda window: free_energy(parameters, window))(window_batch)
        negative = jax.vmap(lambda two: jax.vmap(lambda window: free_energy(parameters, window))(two))(corrupt_batch)
        ranking = jnp.mean(jax.nn.softplus(0.25 + positive[:, None] - negative))
        return ranking + 0.25 * _batch_loss(parameters, pair_batch, negative_batch, lexical_config, jax, jnp)

    def step(parameters, optimizer_state, window_batch, corrupt_batch, pair_batch, negative_batch):
        loss, gradients = jax.value_and_grad(joint_loss)(parameters, window_batch, corrupt_batch, pair_batch, negative_batch)
        updates, next_state = optimizer.update(gradients, optimizer_state, parameters)
        return optax.apply_updates(parameters, updates), next_state, loss

    compiled_step = jax.jit(step)
    best_loss: float | None = None
    best_step = 0
    best_raw = _copy(raw)
    for update in range(config.joint_updates):
        window_ids = data["joint_schedule"][update % len(data["joint_schedule"])]
        window_ids = window_ids[: config.joint_batch] if config.joint_batch <= 4 else np.resize(window_ids, config.joint_batch)
        pair_ids = data["lexical_schedule"][update % len(data["lexical_schedule"])]
        pair_ids = pair_ids[: config.lexical_batch] if config.lexical_batch <= 64 else np.resize(pair_ids, config.lexical_batch)
        raw, state, _ = compiled_step(raw, state, windows[window_ids], corruptions[window_ids], pairs[pair_ids], negatives[pair_ids])
        step_number = update + 1
        if step_number % config.validation_every == 0 or step_number == config.joint_updates:
            value = float(joint_loss(raw, validation_windows, validation_corruptions, validation_pairs, validation_negatives))
            if best_loss is None or value < best_loss:
                best_loss, best_step, best_raw = value, step_number, _copy(raw)
    final_raw = best_raw
    lambda_zero = float(free_energy({name: jnp.asarray(value) for name, value in final_raw.items()}, jnp.asarray(data["train_windows"][0]), 0.0))
    metrics = {
        "protocol.sha256": data["protocol_sha256"],
        "schedule.sha256": data["schedule_sha256"],
        "selection.lexical_step": lexical_metrics["selection.lexical_step"],
        "selection.lexical_validation_auc": lexical_metrics["validation.lexical_macro_auc"],
        "selection.joint_step": best_step,
        "validation.joint_total_loss": best_loss if best_loss is not None else float("nan"),
        "numerics.free_energy_lambda0_abs": abs(lambda_zero),
        "quadrature.order": config.quadrature_order,
        "quadrature.terms_per_window": 8 * config.quadrature_order**3,
        "training.stage": "gaussian-joint quadrature precheck; no sense-region refinement or sampler audit",
    }
    metrics.update(quadrature_refinement_audit(final_raw, data["train_windows"][0], config))
    return final_raw, metrics
