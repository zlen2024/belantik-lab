"""Exact analytic-overlap lexical trainer for the frozen Gaussian baseline.

This is the lexical stage only: a D=1 normalised mixture per word, trained
with the same synthetic pairs, negatives, schedules and validation selection
as the Ising pilot.  The frozen joint reference starts at M=2; the lexical
capacity ablation also permits M=1 and M=4.  It intentionally does not claim
joint-window equilibrium or thermodynamic-hardware execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from word_energy.ising.exact_training import prepare_data
from word_energy.synthetic import POOLS, vocabulary


def _imports():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import optax

    return jax, jnp, optax


@dataclass(frozen=True)
class GaussianLexicalConfig:
    seed: int = 17
    dimension: int = 1
    components: int = 2
    updates: int = 400
    batch_size: int = 64
    learning_rate: float = 0.01
    clip_norm: float = 5.0
    validation_every: int = 20
    mean_norm_bound: float = 3.0
    variance_floor: float = 0.16
    variance_span: float = 1.28


def _sigmoid(value, jnp):
    return 1.0 / (1.0 + jnp.exp(-value))


def _constrained(raw, config: GaussianLexicalConfig, jnp):
    means = raw["means"]
    norms = jnp.linalg.norm(means, axis=-1, keepdims=True)
    means = means * jnp.minimum(1.0, config.mean_norm_bound / jnp.maximum(norms, 1e-12))
    return {
        "logits": raw["logits"],
        "means": means,
        "variances": config.variance_floor + config.variance_span * _sigmoid(raw["raw_variances"], jnp),
    }


def _log_overlap(parameters, left, right, jnp):
    """log integral p_left(z) p_right(z) dz for diagonal Gaussian mixtures."""
    means_left, means_right = parameters["means"][left], parameters["means"][right]
    covariance = parameters["variances"][left][:, None, :] + parameters["variances"][right][None, :, :]
    delta = means_left[:, None, :] - means_right[None, :, :]
    dimension = means_left.shape[-1]
    component_log = -0.5 * (
        dimension * jnp.log(2.0 * jnp.pi)
        + jnp.sum(jnp.log(covariance), axis=-1)
        + jnp.sum(delta * delta / covariance, axis=-1)
    )
    log_weight_left = jax_log_softmax(parameters["logits"][left], jnp)[:, None]
    log_weight_right = jax_log_softmax(parameters["logits"][right], jnp)[None, :]
    return jax_logsumexp(component_log + log_weight_left + log_weight_right, jnp)


def jax_log_softmax(values, jnp):
    return values - jax_logsumexp(values, jnp)


def jax_logsumexp(values, jnp):
    maximum = jnp.max(values)
    return maximum + jnp.log(jnp.sum(jnp.exp(values - maximum)))


def _compatibility_energy(raw, left, right, config: GaussianLexicalConfig, jnp):
    return -_log_overlap(_constrained(raw, config, jnp), left, right, jnp)


def _batch_loss(raw, pair_batch, negative_batch, config: GaussianLexicalConfig, jax, jnp):
    positive = jax.vmap(lambda pair: _compatibility_energy(raw, pair[0], pair[1], config, jnp))(pair_batch)
    negative = jax.vmap(
        lambda left, row: jax.vmap(lambda right: _compatibility_energy(raw, left, right, config, jnp))(row)
    )(pair_batch[:, 0], negative_batch)
    return jnp.mean(jax.nn.softplus(1.0 + positive[:, None] - negative))


def _validation_auc(raw, data: dict[str, Any], config: GaussianLexicalConfig, jnp) -> float:
    ids = data["ids"]
    bank = ids["bank"]
    parameters = _constrained(raw, config, jnp)

    def score(token: int) -> float:
        return float(_log_overlap(parameters, bank, token, jnp))

    by_family: dict[str, list[float]] = {"F": [], "R": []}
    for window, family in zip(data["validation_windows"], data["validation_families"]):
        if family in by_family:
            by_family[family].extend(score(int(token)) for token in window[1:])
    music = [score(ids[token]) for token in POOLS["M"]]

    def auc(positive: list[float]) -> float:
        return float(np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in positive for b in music]))

    return float(np.mean([auc(by_family["F"]), auc(by_family["R"])]))


def _is_better(candidate: tuple[float, float, int], current: tuple[float, float, int] | None) -> bool:
    if current is None:
        return True
    score, loss, step = candidate
    best_score, best_loss, best_step = current
    return score > best_score or (score == best_score and (loss < best_loss or (loss == best_loss and step < best_step)))


def _copy(raw) -> dict[str, np.ndarray]:
    return {name: np.asarray(value).copy() for name, value in raw.items()}


def train_lexical(config: GaussianLexicalConfig = GaussianLexicalConfig()) -> tuple[dict[str, np.ndarray], dict[str, float | int | str]]:
    """Train the preregistered lexical-only Gaussian baseline with exact overlap."""
    if config.dimension != 1 or config.components not in {1, 2, 4}:
        raise ValueError("The lexical capacity ablation is restricted to D=1 and M in {1, 2, 4}.")
    jax, jnp, optax = _imports()
    data = prepare_data()
    key = jax.random.key(config.seed)
    key, km, kl = jax.random.split(key, 3)
    vocab_size = len(vocabulary())
    variance_probability = (0.36 - config.variance_floor) / config.variance_span
    raw = {
        "means": jax.random.normal(km, (vocab_size, config.components, config.dimension), dtype=jnp.float64) * 0.8,
        "raw_variances": jnp.full(
            (vocab_size, config.components, config.dimension), jnp.log(variance_probability / (1.0 - variance_probability)), dtype=jnp.float64
        ),
        "logits": jax.random.normal(kl, (vocab_size, config.components), dtype=jnp.float64) * 0.1,
    }
    optimizer = optax.chain(optax.clip_by_global_norm(config.clip_norm), optax.adam(config.learning_rate))
    state = optimizer.init(raw)
    pairs = jnp.asarray(data["train_pairs"])
    negatives = jnp.asarray(data["train_negatives"])
    validation_pairs = jnp.asarray(data["validation_pairs"])
    validation_negatives = jnp.asarray(data["validation_negatives"])

    def step(parameters, optimizer_state, pair_batch, negative_batch):
        loss, gradients = jax.value_and_grad(_batch_loss)(parameters, pair_batch, negative_batch, config, jax, jnp)
        updates, next_state = optimizer.update(gradients, optimizer_state, parameters)
        return optax.apply_updates(parameters, updates), next_state, loss

    compiled_step = jax.jit(step)
    best: tuple[float, float, int] | None = None
    best_raw = _copy(raw)
    for update in range(config.updates):
        schedule = data["lexical_schedule"][update % len(data["lexical_schedule"])]
        indices = schedule[: config.batch_size] if config.batch_size <= 64 else np.resize(schedule, config.batch_size)
        raw, state, _ = compiled_step(raw, state, pairs[indices], negatives[indices])
        step_number = update + 1
        if step_number % config.validation_every == 0 or step_number == config.updates:
            validation_loss = float(_batch_loss(raw, validation_pairs, validation_negatives, config, jax, jnp))
            validation_auc = _validation_auc(raw, data, config, jnp)
            candidate = (validation_auc, validation_loss, step_number)
            if _is_better(candidate, best):
                best, best_raw = candidate, _copy(raw)
    return best_raw, {
        "protocol.sha256": data["protocol_sha256"],
        "schedule.sha256": data["schedule_sha256"],
        "selection.lexical_step": best[2] if best else 0,
        "validation.lexical_macro_auc": best[0] if best else float("nan"),
        "validation.lexical_loss": best[1] if best else float("nan"),
        "training.stage": f"gaussian-lexical-only precheck; D=1 M={config.components}",
    }


def evaluate_lexical(raw: dict[str, np.ndarray], split: str = "test") -> dict[str, float]:
    """Evaluate bank F/R versus M retrieval using an explicitly selected split."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    _, jnp, _ = _imports()
    data = prepare_data()
    config = GaussianLexicalConfig()
    parameters = _constrained(raw, config, jnp)
    ids = data["ids"]

    def score(token: int) -> float:
        return float(_log_overlap(parameters, ids["bank"], token, jnp))

    values: dict[str, list[float]] = {"F": [], "R": [], "M": [score(ids[token]) for token in POOLS["M"]]}
    for window, family in zip(data[f"{split}_windows"], data[f"{split}_families"]):
        if family in {"F", "R"}:
            values[family].extend(score(int(token)) for token in window[1:])

    def auc(positive: list[float]) -> float:
        return float(np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in positive for b in values["M"]]))

    f_auc, r_auc = auc(values["F"]), auc(values["R"])
    return {
        f"{split}.lexical_auc_f_vs_m": f_auc,
        f"{split}.lexical_auc_r_vs_m": r_auc,
        f"{split}.lexical_macro_auc": float(np.mean([f_auc, r_auc])),
    }
