import numpy as np

from word_energy.energy_embedding.mcmc_diagnostics import split_rank_rhat_ess


def test_split_rank_diagnostics_recognise_iid_draws() -> None:
    values = np.random.default_rng(7).normal(size=(4, 1_000))
    rhat, ess, mcse = split_rank_rhat_ess(values)
    assert rhat < 1.05
    assert ess > 400
    assert mcse < 0.1
