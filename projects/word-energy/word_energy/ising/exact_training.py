"""Exact-free-energy ranking trainer for the frozen two-spin Ising pilot.

The active three-token graph has only 64 states. Training therefore uses exact
enumeration and autodiff through ``logsumexp``; it is neither CD/PCD nor a
gradient through discrete Gibbs samples. Data construction, corruptions and
batch schedules are deterministic protocol objects, independent of model seed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from word_energy.synthetic import POOLS, split_windows, vocabulary


def _imports():
    import jax

    # The frozen pilot specifies float64; set it before importing jax.numpy.
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import optax

    return jax, jnp, optax


@dataclass(frozen=True)
class ExactTrainingConfig:
    seed: int = 17
    lexical_updates: int = 400
    joint_updates: int = 100
    lexical_batch: int = 64
    joint_batch: int = 4
    lexical_lr: float = 0.01
    joint_lr: float = 0.003
    clip_norm: float = 5.0
    validation_every: int = 20
    permuted_lexical_contexts: bool = False


def token_index() -> dict[str, int]:
    return {token: index for index, token in enumerate(vocabulary())}


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _spin_states(jnp, count: int):
    indices = jnp.arange(1 << count, dtype=jnp.int32)
    return (2 * ((indices[:, None] >> jnp.arange(count)) & 1) - 1).astype(jnp.float64)


def _parameters(raw, jnp):
    return {
        "biases": 2.5 * jnp.tanh(raw["biases"]),
        "internal": 2.0 * jnp.tanh(raw["internal"]),
        "interaction": 1.5 * jax_nn_sigmoid(raw["interaction"], jnp),
    }


def jax_nn_sigmoid(values, jnp):
    return 1.0 / (1.0 + jnp.exp(-values))


def _free_energy_factory():
    jax, jnp, _ = _imports()
    unary_states = _spin_states(jnp, 2)
    pair_states = _spin_states(jnp, 4).reshape(-1, 2, 2)
    triple_states = _spin_states(jnp, 6).reshape(-1, 3, 2)

    def unary_energy(parameters, token, states):
        return -(
            jnp.sum(parameters["biases"][token] * states, axis=-1)
            + parameters["internal"][token] * states[..., 0] * states[..., 1]
        )

    def unary_log_z(parameters, token):
        return jax.scipy.special.logsumexp(-unary_energy(parameters, token, unary_states))

    def unary_probability(raw, token):
        parameters = _parameters(raw, jnp)
        log_probability = -unary_energy(parameters, token, unary_states) - unary_log_z(parameters, token)
        return jnp.exp(log_probability)

    def window_free_energy(raw, tokens, coupling_scale: float = 1.0):
        parameters = _parameters(raw, jnp)
        states = pair_states if tokens.shape[0] == 2 else triple_states
        energy = jnp.zeros(states.shape[0], dtype=jnp.float64)
        for position in range(tokens.shape[0]):
            energy = energy + unary_energy(parameters, tokens[position], states[:, position])
        for position in range(1, tokens.shape[0]):
            energy = energy - coupling_scale * jnp.sum(
                parameters["interaction"] * states[:, 0] * states[:, position], axis=-1
            )
        return -jax.scipy.special.logsumexp(-energy) + sum(
            unary_log_z(parameters, tokens[position]) for position in range(tokens.shape[0])
        )

    def joint_center_probability(raw, tokens, coupling_scale: float = 1.0):
        """Exact marginal over the centre token of a three-token window."""
        parameters = _parameters(raw, jnp)
        energy = jnp.zeros(triple_states.shape[0], dtype=jnp.float64)
        for position in range(3):
            energy = energy + unary_energy(parameters, tokens[position], triple_states[:, position])
        for position in range(1, 3):
            energy = energy - coupling_scale * jnp.sum(
                parameters["interaction"] * triple_states[:, 0] * triple_states[:, position], axis=-1
            )
        weights = jax.nn.softmax(-energy)
        return jnp.asarray(
            [jnp.sum(weights * jnp.all(triple_states[:, 0] == state, axis=-1)) for state in unary_states]
        )

    return jax, jnp, window_free_energy, unary_probability, joint_center_probability


def _valid_grammar_pairs(ids: dict[str, int]) -> set[tuple[int, int]]:
    """All ordered token pairs permitted by the synthetic grammar, not model scores."""
    valid: set[tuple[int, int]] = set()
    for family, anchors in POOLS.items():
        center = ids["song"] if family == "M" else ids["bank"]
        members = [center, *(ids[token] for token in anchors)]
        valid.update((left, right) for left in members for right in members if left != right)
    return valid


def _ordered_pairs(windows: np.ndarray) -> np.ndarray:
    return np.asarray(
        [(row[left], row[right]) for row in windows for left in range(3) for right in range(3) if left != right],
        dtype=np.int32,
    )


def _frequency(ids_windows: np.ndarray, vocab_size: int) -> np.ndarray:
    counts = np.bincount(ids_windows.reshape(-1), minlength=vocab_size).astype(np.float64)
    return counts**0.75 / np.sum(counts**0.75)


def _negative_table(
    pairs: np.ndarray,
    probabilities: np.ndarray,
    valid_pairs: set[tuple[int, int]],
    *,
    seed: int,
) -> np.ndarray:
    """Draw five grammar-filtered unigram^.75 negatives per canonical pair."""
    rng = np.random.default_rng(seed)
    choices = np.arange(len(probabilities), dtype=np.int32)
    result = np.empty((len(pairs), 5), dtype=np.int32)
    for row, (left, right) in enumerate(pairs):
        allowed = np.asarray(
            [candidate for candidate in choices if candidate != left and candidate != right and (int(left), int(candidate)) not in valid_pairs],
            dtype=np.int32,
        )
        allowed_weights = probabilities[allowed]
        allowed_weights = allowed_weights / allowed_weights.sum()
        result[row] = rng.choice(allowed, size=5, replace=True, p=allowed_weights)
    return result


def _corruptions(windows: np.ndarray, families: tuple[str, ...], ids: dict[str, int]) -> np.ndarray:
    """Create two fixed cross-pool anchor replacements per canonical window."""
    rng = np.random.default_rng(202)
    corrupted = np.empty((len(windows), 2, 3), dtype=np.int32)
    pool_names = tuple(POOLS)
    for row, (window, family) in enumerate(zip(windows, families)):
        alternatives = tuple(name for name in pool_names if name != family)
        for corruption_index, slot in enumerate((1, 2)):
            replacement_pool = alternatives[rng.integers(0, len(alternatives))]
            replacement = POOLS[replacement_pool][rng.integers(0, len(POOLS[replacement_pool]))]
            corrupted[row, corruption_index] = window
            corrupted[row, corruption_index, slot] = ids[replacement]
    return corrupted


def prepare_data() -> dict[str, Any]:
    """Return the frozen corpus-derived protocol, including batch IDs and hashes."""
    ids = token_index()
    splits = split_windows()
    rows = {
        name: np.asarray([[ids[token] for token in row.tokens] for row in values], dtype=np.int32)
        for name, values in splits.items()
    }
    families = {name: tuple(row.family for row in values) for name, values in splits.items()}
    train_pairs = _ordered_pairs(rows["train"])
    validation_pairs = _ordered_pairs(rows["validation"])
    probabilities = _frequency(rows["train"], len(ids))
    valid_pairs = _valid_grammar_pairs(ids)
    train_negatives = _negative_table(train_pairs, probabilities, valid_pairs, seed=6060)
    validation_negatives = _negative_table(validation_pairs, probabilities, valid_pairs, seed=7070)
    lexical_schedule = np.random.default_rng(4040).choice(len(train_pairs), size=(400, 64), replace=True).astype(np.int32)
    joint_schedule = np.random.default_rng(5050).choice(len(rows["train"]), size=(100, 4), replace=True).astype(np.int32)
    permuted_train_pairs = train_pairs.copy()
    permuted_train_pairs[:, 1] = np.random.default_rng(8080).permutation(permuted_train_pairs[:, 1])
    result: dict[str, Any] = {
        "train_windows": rows["train"],
        "validation_windows": rows["validation"],
        "test_windows": rows["test"],
        "train_families": families["train"],
        "validation_families": families["validation"],
        "test_families": families["test"],
        "train_pairs": train_pairs,
        "permuted_train_pairs": permuted_train_pairs,
        "validation_pairs": validation_pairs,
        "train_negatives": train_negatives,
        "validation_negatives": validation_negatives,
        "train_corruptions": _corruptions(rows["train"], families["train"], ids),
        "validation_corruptions": _corruptions(rows["validation"], families["validation"], ids),
        "lexical_schedule": lexical_schedule,
        "joint_schedule": joint_schedule,
        "ids": ids,
    }
    frozen = {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in result.items()
        if name not in {"ids", "permuted_train_pairs"}
    }
    result["protocol_sha256"] = _sha256(frozen)
    result["schedule_sha256"] = _sha256(
        {"lexical_schedule": lexical_schedule.tolist(), "joint_schedule": joint_schedule.tolist()}
    )
    return result


def _lexical_auc(raw, free_energy, data, jnp) -> float:
    """Macro pairwise AUC: bank-F vs M and bank-R vs M on validation anchors."""
    ids = data["ids"]
    bank, music = ids["bank"], [ids[token] for token in POOLS["M"]]
    scores: dict[str, list[float]] = {"F": [], "R": [], "M": []}
    for window, actual_family in zip(data["validation_windows"], data["validation_families"]):
        if actual_family in {"F", "R"}:
            scores[actual_family].extend(
                -float(free_energy(raw, jnp.asarray([bank, anchor], dtype=jnp.int32))) for anchor in window[1:]
            )
    scores["M"] = [
        -float(free_energy(raw, jnp.asarray([bank, anchor], dtype=jnp.int32))) for anchor in music
    ]

    def auc(positive: list[float]) -> float:
        comparisons = [1.0 if a > b else 0.5 if a == b else 0.0 for a in positive for b in scores["M"]]
        return float(np.mean(comparisons))

    return float(np.mean([auc(scores["F"]), auc(scores["R"])]))


def _sense_accuracy_and_loss(raw, free_energy, unary_probability, center_probability, data, jnp) -> tuple[float, float]:
    """Validation map uses train anchors only, then scores bank validation windows."""
    by_family: dict[str, list[int]] = {"F": [], "R": []}
    for window, family in zip(data["train_windows"], data["train_families"]):
        if family in by_family:
            by_family[family].extend(window[1:].tolist())
    q = {
        family: np.mean([np.asarray(unary_probability(raw, token)) for token in anchors], axis=0)
        for family, anchors in by_family.items()
    }
    region_f = q["F"] > q["R"]
    equal = q["F"] == q["R"]
    credits: dict[str, list[float]] = {"F": [], "R": []}
    for window, family in zip(data["validation_windows"], data["validation_families"]):
        if family not in credits:
            continue
        mass = np.asarray(center_probability(raw, jnp.asarray(window, dtype=jnp.int32)))
        f_mass = float(mass[region_f].sum() + 0.5 * mass[equal].sum())
        r_mass = 1.0 - f_mass
        correct = f_mass > r_mass if family == "F" else r_mass > f_mass
        tied = f_mass == r_mass
        credits[family].append(1.0 if correct else 0.5 if tied else 0.0)
    macro_accuracy = float(np.mean([np.mean(credits["F"]), np.mean(credits["R"])]))
    windows = jnp.asarray(data["validation_windows"])
    corruptions = jnp.asarray(data["validation_corruptions"])
    positive = _batch_energy(free_energy, raw, windows)
    negative = _batch_energy(free_energy, raw, corruptions.reshape((-1, 3))).reshape((-1, 2))
    joint_loss = float(jnp.mean(jnp.logaddexp(0.0, 0.25 + positive[:, None] - negative)))
    return macro_accuracy, joint_loss


def _sense_region(raw, unary_probability, data) -> tuple[np.ndarray, np.ndarray]:
    """Return the train-anchor F region and the equality states (half credit)."""
    anchors: dict[str, list[int]] = {"F": [], "R": []}
    for window, family in zip(data["train_windows"], data["train_families"]):
        if family in anchors:
            anchors[family].extend(window[1:].tolist())
    q_f = np.mean([np.asarray(unary_probability(raw, token)) for token in anchors["F"]], axis=0)
    q_r = np.mean([np.asarray(unary_probability(raw, token)) for token in anchors["R"]], axis=0)
    return q_f > q_r, q_f == q_r


def _region_mass(probability: np.ndarray, region_f: np.ndarray, equal: np.ndarray) -> float:
    return float(probability[region_f].sum() + 0.5 * probability[equal].sum())


def evaluate_checkpoint(raw: dict[str, np.ndarray], split: str = "test") -> dict[str, float]:
    """Evaluate a fixed checkpoint without using labels as inference inputs.

    The map is derived only from train anchors. ``split`` is explicit so callers
    cannot accidentally hide a holdout evaluation behind training code.
    """
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    _, jnp, free_energy, unary_probability, center_probability = _free_energy_factory()
    data = prepare_data()
    region_f, equal = _sense_region(raw, unary_probability, data)
    ids = data["ids"]
    windows = data[f"{split}_windows"]
    families = data[f"{split}_families"]
    credits: dict[str, list[float]] = {"F": [], "R": []}
    conditional_mass: list[float] = []
    prior_mass: list[float] = []
    lambda0_mass: list[float] = []
    bank_prior = np.asarray(unary_probability(raw, ids["bank"]))
    for window, family in zip(windows, families):
        if family not in credits:
            continue
        marginal = np.asarray(center_probability(raw, jnp.asarray(window, dtype=jnp.int32), 1.0))
        zero_marginal = np.asarray(center_probability(raw, jnp.asarray(window, dtype=jnp.int32), 0.0))
        f_mass = _region_mass(marginal, region_f, equal)
        f_prior = _region_mass(bank_prior, region_f, equal)
        f_zero = _region_mass(zero_marginal, region_f, equal)
        true_mass = f_mass if family == "F" else 1.0 - f_mass
        true_prior = f_prior if family == "F" else 1.0 - f_prior
        true_zero = f_zero if family == "F" else 1.0 - f_zero
        other_mass = 1.0 - true_mass
        credits[family].append(1.0 if true_mass > other_mass else 0.5 if true_mass == other_mass else 0.0)
        conditional_mass.append(true_mass)
        prior_mass.append(true_prior)
        lambda0_mass.append(true_zero)

    def pairwise_auc(positive_tokens: tuple[str, ...]) -> float:
        bank = ids["bank"]
        positive = [-float(free_energy(raw, jnp.asarray([bank, ids[token]], dtype=jnp.int32), 1.0)) for token in positive_tokens]
        negative = [-float(free_energy(raw, jnp.asarray([bank, ids[token]], dtype=jnp.int32), 1.0)) for token in POOLS["M"]]
        return float(np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in positive for b in negative]))

    auc_f = pairwise_auc(POOLS["F"])
    auc_r = pairwise_auc(POOLS["R"])
    return {
        f"{split}.macro_sense_accuracy": float(np.mean([np.mean(credits["F"]), np.mean(credits["R"])])),
        f"{split}.correct_region_posterior": float(np.mean(conditional_mass)),
        f"{split}.correct_region_prior": float(np.mean(prior_mass)),
        f"{split}.correct_region_gain_lambda1_minus_lambda0": float(np.mean(conditional_mass) - np.mean(lambda0_mass)),
        f"{split}.lexical_auc_f_vs_m": auc_f,
        f"{split}.lexical_auc_r_vs_m": auc_r,
        f"{split}.lexical_macro_auc": float(np.mean([auc_f, auc_r])),
    }


def _batch_energy(free_energy, raw, windows):
    jax, _, _ = _imports()
    return jax.vmap(lambda window: free_energy(raw, window))(windows)


def _is_better(candidate: tuple[float, float, int], current: tuple[float, float, int] | None) -> bool:
    """Higher primary metric, then lower loss, then earlier checkpoint."""
    if current is None:
        return True
    score, loss, step = candidate
    best_score, best_loss, best_step = current
    return score > best_score or (score == best_score and (loss < best_loss or (loss == best_loss and step < best_step)))


def _copy_raw(raw) -> dict[str, np.ndarray]:
    return {name: np.asarray(value).copy() for name, value in raw.items()}


def train_smoke(config: ExactTrainingConfig = ExactTrainingConfig(), *, snapshot_callback=None, branch: str = 'original', initial_override=None) -> tuple[dict[str, np.ndarray], dict[str, float | int | str]]:
    """Run the frozen protocol (or a shorter smoke subset) and select checkpoints."""
    jax, jnp, optax = _imports()
    _, _, free_energy, unary_probability, center_probability = _free_energy_factory()
    data = prepare_data()
    vocab_size = len(vocabulary())
    key = jax.random.key(config.seed)
    key, kb, kj = jax.random.split(key, 3)
    raw = {
        "biases": jax.random.normal(kb, (vocab_size, 2), dtype=jnp.float64) * 0.1,
        "internal": jax.random.normal(kj, (vocab_size,), dtype=jnp.float64) * 0.1,
        "interaction": jnp.full((2,), jnp.log(0.8 / (1.5 - 0.8)), dtype=jnp.float64),
    }
    if initial_override is not None:
        raw = {name: jnp.asarray(value) for name, value in initial_override.items()}
    if branch not in {'original', 'joint_from_initial', 'lexical_auxiliary'}:
        raise ValueError('unknown diagnostic branch')
    if snapshot_callback is not None:
        snapshot_callback('initial', 0, _copy_raw(raw))
    initial_raw = _copy_raw(raw)
    lexical_optimizer = optax.chain(optax.clip_by_global_norm(config.clip_norm), optax.adam(config.lexical_lr))
    lexical_state = lexical_optimizer.init(raw)
    train_pairs = jnp.asarray(data["permuted_train_pairs"] if config.permuted_lexical_contexts else data["train_pairs"])
    train_negatives = jnp.asarray(data["train_negatives"])
    validation_pairs = jnp.asarray(data["validation_pairs"])
    validation_negatives = jnp.asarray(data["validation_negatives"])

    def lexical_loss(parameters, pair_batch, negative_batch):
        positive = jax.vmap(lambda pair: free_energy(parameters, pair))(pair_batch)
        negative = jax.vmap(
            lambda left, row: jax.vmap(
                lambda right: free_energy(parameters, jnp.asarray([left, right], dtype=jnp.int32))
            )(row)
        )(pair_batch[:, 0], negative_batch)
        return jnp.mean(jax.nn.softplus(1.0 + positive[:, None] - negative))

    lexical_step = jax.jit(
        lambda parameters, state, pair_batch, negative_batch: _optimizer_step(
            lexical_loss, lexical_optimizer, parameters, state, pair_batch, negative_batch
        )
    )
    lexical_best: tuple[float, float, int] | None = None
    lexical_raw = _copy_raw(raw)
    for update in range(config.lexical_updates):
        schedule_row = data["lexical_schedule"][update % len(data["lexical_schedule"])]
        batch_ids = schedule_row[: config.lexical_batch] if config.lexical_batch <= 64 else np.resize(schedule_row, config.lexical_batch)
        raw, lexical_state, _ = lexical_step(raw, lexical_state, train_pairs[batch_ids], train_negatives[batch_ids])
        step = update + 1
        if step % config.validation_every == 0 or step == config.lexical_updates:
            if snapshot_callback is not None:
                snapshot_callback('lexical', step, _copy_raw(raw))
            validation_loss = float(lexical_loss(raw, validation_pairs, validation_negatives))
            validation_auc = _lexical_auc(raw, free_energy, data, jnp)
            candidate = (validation_auc, validation_loss, step)
            if _is_better(candidate, lexical_best):
                lexical_best, lexical_raw = candidate, _copy_raw(raw)
    raw = {name: jnp.asarray(value) for name, value in lexical_raw.items()}
    if snapshot_callback is not None:
        snapshot_callback('lexical_selected', lexical_best[2] if lexical_best else 0, _copy_raw(raw))
    if branch == 'joint_from_initial':
        raw = {name: jnp.asarray(value) for name, value in initial_raw.items()}

    joint_optimizer = optax.chain(optax.clip_by_global_norm(config.clip_norm), optax.adam(config.joint_lr))
    joint_state = joint_optimizer.init(raw)
    windows = jnp.asarray(data["train_windows"])
    corruptions = jnp.asarray(data["train_corruptions"])

    def joint_loss(parameters, window_batch, corrupt_batch, auxiliary_pair_batch, auxiliary_negative_batch):
        positives = jax.vmap(lambda window: free_energy(parameters, window))(window_batch)
        negatives = jax.vmap(
            lambda two: jax.vmap(lambda window: free_energy(parameters, window))(two)
        )(corrupt_batch)
        ranking = jnp.mean(jax.nn.softplus(0.25 + positives[:, None] - negatives))
        auxiliary = 0.25 * lexical_loss(parameters, auxiliary_pair_batch, auxiliary_negative_batch)
        return auxiliary if branch == 'lexical_auxiliary' else ranking + auxiliary

    joint_step = jax.jit(
        lambda parameters, state, window_batch, corrupt_batch, pair_batch, negative_batch: _optimizer_step(
            joint_loss, joint_optimizer, parameters, state, window_batch, corrupt_batch, pair_batch, negative_batch
        )
    )
    joint_best: tuple[float, float, int] | None = None
    joint_raw = _copy_raw(raw)
    for update in range(config.joint_updates):
        schedule_row = data["joint_schedule"][update % len(data["joint_schedule"])]
        window_ids = schedule_row[: config.joint_batch] if config.joint_batch <= 4 else np.resize(schedule_row, config.joint_batch)
        auxiliary_ids = data["lexical_schedule"][update % len(data["lexical_schedule"])]
        auxiliary_ids = auxiliary_ids[: config.lexical_batch] if config.lexical_batch <= 64 else np.resize(auxiliary_ids, config.lexical_batch)
        raw, joint_state, _ = joint_step(
            raw,
            joint_state,
            windows[window_ids],
            corruptions[window_ids],
            train_pairs[auxiliary_ids],
            train_negatives[auxiliary_ids],
        )
        step = update + 1
        if step % config.validation_every == 0 or step == config.joint_updates:
            if snapshot_callback is not None:
                snapshot_callback('joint', step, _copy_raw(raw))
            sense_accuracy, joint_validation_loss = _sense_accuracy_and_loss(
                raw, free_energy, unary_probability, center_probability, data, jnp
            )
            candidate = (sense_accuracy, joint_validation_loss, step)
            if _is_better(candidate, joint_best):
                joint_best, joint_raw = candidate, _copy_raw(raw)
    final_raw = joint_raw
    if snapshot_callback is not None:
        snapshot_callback('joint_selected', joint_best[2] if joint_best else 0, _copy_raw(final_raw))
    params = _parameters({name: jnp.asarray(value) for name, value in final_raw.items()}, jnp)
    metrics: dict[str, float | int | str] = {
        "protocol.sha256": data["protocol_sha256"],
        "schedule.sha256": data["schedule_sha256"],
        "selection.lexical_step": lexical_best[2] if lexical_best else 0,
        "selection.lexical_validation_auc": lexical_best[0] if lexical_best else float("nan"),
        "selection.lexical_validation_loss": lexical_best[1] if lexical_best else float("nan"),
        "selection.joint_step": joint_best[2] if joint_best else 0,
        "validation.macro_sense_accuracy": joint_best[0] if joint_best else float("nan"),
        "validation.joint_ranking_loss": joint_best[1] if joint_best else float("nan"),
        "interaction.weight_0": float(params["interaction"][0]),
        "interaction.weight_1": float(params["interaction"][1]),
        "training.stage": (
            "permuted-lexical-context control"
            if config.permuted_lexical_contexts
            else "frozen-budget precheck"
            if (config.lexical_updates, config.joint_updates) == (400, 100)
            else "short smoke"
        ),
    }
    return final_raw, metrics


def _optimizer_step(loss_function, optimizer, parameters, state, *batch):
    jax, _, optax = _imports()
    loss, gradients = jax.value_and_grad(loss_function)(parameters, *batch)
    updates, next_state = optimizer.update(gradients, state, parameters)
    return optax.apply_updates(parameters, updates), next_state, loss
