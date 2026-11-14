"""Minimal split-chain diagnostics for the Gaussian sampler audit."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


def split_rank_rhat_ess(values: np.ndarray) -> tuple[float, float, float]:
    """Return rank-normalised split R-hat, ESS and MCSE for ``[chain, draw]``."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 4:
        raise ValueError("Need values[at least two chains, at least four draws]")
    half = values.shape[1] // 2
    split = np.concatenate((values[:, :half], values[:, -half:]), axis=0)
    ranked = rankdata(split.reshape(-1), method="average").reshape(split.shape)
    normalised = norm.ppf((ranked - 0.5) / ranked.size)
    within = np.mean(np.var(normalised, axis=1, ddof=1))
    if within <= 1e-15:
        return float("inf"), 0.0, float("inf")
    draws, chains = normalised.shape[1], normalised.shape[0]
    between = draws * np.var(np.mean(normalised, axis=1), ddof=1)
    variance_plus = (draws - 1) * within / draws + between / draws
    rhat = float(np.sqrt(variance_plus / within))
    autocovariances = []
    for lag in range(draws):
        centered = normalised - np.mean(normalised, axis=1, keepdims=True)
        autocovariances.append(float(np.mean(np.mean(centered[:, : draws - lag] * centered[:, lag:], axis=1))))
    correlations = [1.0]
    for lag in range(1, draws):
        correlations.append(1.0 - (within - autocovariances[lag]) / variance_plus)
    tau = 1.0
    for lag in range(1, draws - 1, 2):
        pair = correlations[lag] + correlations[lag + 1]
        if pair <= 0.0:
            break
        tau += 2.0 * pair
    ess = float(min(chains * draws, chains * draws / max(tau, 1.0)))
    mcse = float(np.std(values.reshape(-1), ddof=1) / np.sqrt(ess)) if ess > 0.0 else float("inf")
    return rhat, ess, mcse
