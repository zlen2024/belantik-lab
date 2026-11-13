import numpy as np

from word_energy.energy_embedding.landscape import GaussianMixtureLandscape
from word_energy.energy_embedding.sampler import sample_joint_independence_mh
from word_energy.energy_embedding.system import ActiveWindowSystem


def test_unary_independence_sampler_accepts_every_proposal_at_lambda_zero() -> None:
    landscape = GaussianMixtureLandscape(
        logits=np.zeros((3, 1)), means=np.zeros((3, 1, 1)), variances=np.ones((3, 1, 1))
    )
    result = sample_joint_independence_mh(
        ActiveWindowSystem(landscape, raw_metric=np.ones((1, 1)), coupling=0.0),
        np.array([0, 1, 2]),
        chains=2,
        burn_in_sweeps=2,
        sample_sweeps=3,
        seed=7,
    )
    assert result.samples.shape == (2, 3, 3, 1)
    assert result.acceptance_rate == 1.0
