"""Frozen-budget SGNS contextual-vector baseline for the synthetic Word Energy toy."""

from __future__ import annotations

from dataclasses import dataclass

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
class ContextualVectorConfig:
    seed: int = 17
    dimension: int = 6
    tied: bool = True
    updates: int = 400
    batch_size: int = 64
    learning_rate: float = 0.01
    clip_norm: float = 5.0
    validation_every: int = 20

    @property
    def variant(self) -> str:
        return "tied-d6" if self.tied else "untied-d3"


def _parameters(raw, config: ContextualVectorConfig):
    return raw["input"], raw["input"] if config.tied else raw["output"]


def _scores(raw, left, right, config: ContextualVectorConfig, jnp):
    input_table, output_table = _parameters(raw, config)
    return jnp.sum(input_table[left] * output_table[right], axis=-1)


def _batch_loss(raw, pair_batch, negative_batch, config: ContextualVectorConfig, jax, jnp):
    positive = _scores(raw, pair_batch[:, 0], pair_batch[:, 1], config, jnp)
    negative = jax.vmap(lambda left, row: _scores(raw, left, row, config, jnp))(pair_batch[:, 0], negative_batch)
    return jnp.mean(jax.nn.softplus(-positive) + jnp.sum(jax.nn.softplus(negative), axis=1))


def _lexical_auc(raw, data, config: ContextualVectorConfig, jnp) -> float:
    ids = data["ids"]

    def score(token: int) -> float:
        return float(_scores(raw, ids["bank"], token, config, jnp))

    positives: dict[str, list[float]] = {"F": [], "R": []}
    for window, family in zip(data["validation_windows"], data["validation_families"]):
        if family in positives:
            positives[family].extend(score(int(token)) for token in window[1:])
    music = [score(ids[token]) for token in POOLS["M"]]

    def auc(values: list[float]) -> float:
        return float(np.mean([1.0 if value > other else 0.5 if value == other else 0.0 for value in values for other in music]))

    return float(np.mean([auc(positives["F"]), auc(positives["R"])]))


def _is_better(candidate: tuple[float, float, int], current: tuple[float, float, int] | None) -> bool:
    if current is None:
        return True
    score, loss, step = candidate
    best_score, best_loss, best_step = current
    return score > best_score or (score == best_score and (loss < best_loss or (loss == best_loss and step < best_step)))


def _copy(raw) -> dict[str, np.ndarray]:
    return {name: np.asarray(value).copy() for name, value in raw.items()}


def train_contextual_vector(config: ContextualVectorConfig = ContextualVectorConfig()):
    """Train lexical SGNS; contextual labels remain evaluation-only."""
    expected_dimension = 6 if config.tied else 3
    if config.dimension != expected_dimension:
        raise ValueError("The frozen vector baseline is tied D=6 or untied D=3 only")
    jax, jnp, optax = _imports()
    data = prepare_data()
    key = jax.random.key(config.seed)
    key, input_key, output_key = jax.random.split(key, 3)
    vocab_size = len(vocabulary())
    raw = {"input": jax.random.normal(input_key, (vocab_size, config.dimension), dtype=jnp.float64) * 0.1}
    if not config.tied:
        raw["output"] = jax.random.normal(output_key, (vocab_size, config.dimension), dtype=jnp.float64) * 0.1
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
            loss = float(_batch_loss(raw, validation_pairs, validation_negatives, config, jax, jnp))
            auc = _lexical_auc(raw, data, config, jnp)
            candidate = (auc, loss, step_number)
            if _is_better(candidate, best):
                best, best_raw = candidate, _copy(raw)
    return best_raw, {
        "protocol.sha256": data["protocol_sha256"],
        "schedule.sha256": data["schedule_sha256"],
        "selection.lexical_step": best[2] if best else 0,
        "validation.lexical_macro_auc": best[0] if best else float("nan"),
        "validation.lexical_loss": best[1] if best else float("nan"),
        "training.stage": f"contextual-vector lexical SGNS {config.variant}; no joint adaptation",
    }


def evaluate_contextual_vector(raw: dict[str, np.ndarray], config: ContextualVectorConfig, split: str = "validation") -> dict[str, float]:
    """Use input-table context vectors and train-only F/R centroids."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    data = prepare_data()
    vectors = np.asarray(raw["input"], dtype=np.float64)
    anchors: dict[str, list[int]] = {"F": [], "R": []}
    for window, family in zip(data["train_windows"], data["train_families"]):
        if family in anchors:
            anchors[family].extend(window[1:].tolist())
    centroids = {family: np.mean(vectors[token_ids], axis=0) for family, token_ids in anchors.items()}

    def cosine(left: np.ndarray, right: np.ndarray) -> float:
        norm = float(np.linalg.norm(left) * np.linalg.norm(right))
        return 0.0 if norm == 0.0 else float(np.dot(left, right) / norm)

    credits: dict[str, list[float]] = {"F": [], "R": []}
    counts: dict[str, dict[str, int]] = {"F": {"F": 0, "R": 0, "tie": 0}, "R": {"F": 0, "R": 0, "tie": 0}}
    for window, family in zip(data[f"{split}_windows"], data[f"{split}_families"]):
        if family not in credits:
            continue
        context = np.mean(vectors[window[1:]], axis=0)
        state = vectors[window[0]] + context
        f_score, r_score = cosine(state, centroids["F"]), cosine(state, centroids["R"])
        prediction = "F" if f_score > r_score else "R" if f_score < r_score else "tie"
        counts[family][prediction] += 1
        credits[family].append(1.0 if prediction == family else 0.5 if prediction == "tie" else 0.0)
    return {
        f"{split}.context_macro_sense_accuracy": float(np.mean([np.mean(credits["F"]), np.mean(credits["R"])])),
        f"{split}.context_f_correct_count": float(counts["F"]["F"]),
        f"{split}.context_r_correct_count": float(counts["R"]["R"]),
        f"{split}.context_f_predicted_r": float(counts["F"]["R"]),
        f"{split}.context_r_predicted_f": float(counts["R"]["F"]),
        f"{split}.context_tie_count": float(counts["F"]["tie"] + counts["R"]["tie"]),
    }
