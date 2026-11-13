import numpy as np

from word_energy.energy_embedding import (
    ActiveWindowSystem,
    GaussianMixtureLandscape,
    analytic_overlap,
    gaussian_hermite_center_marginal_1d,
    gaussian_hermite_joint_reference,
)
from word_energy.energy_embedding.reference import grid_reference_1d


def _landscape() -> GaussianMixtureLandscape:
    return GaussianMixtureLandscape(
        logits=np.array([[0.0, 0.0], [0.2, -0.2]]),
        means=np.array([[[-2.0], [2.0]], [[2.0], [3.0]]]),
        variances=np.array([[[0.5], [0.5]], [[0.3], [0.7]]]),
    )


def test_unary_density_is_normalised_on_a_wide_grid() -> None:
    landscape = _landscape()
    grid = np.linspace(-12.0, 12.0, 10001)[:, None]
    integral = np.trapezoid(np.exp(landscape.log_density(0, grid)), grid[:, 0])
    assert abs(integral - 1.0) < 1e-6


def test_analytic_overlap_matches_quadrature() -> None:
    landscape = _landscape()
    grid = np.linspace(-12.0, 12.0, 10001)[:, None]
    quadrature = np.trapezoid(
        np.exp(landscape.log_density(0, grid)) * np.exp(landscape.log_density(1, grid)), grid[:, 0]
    )
    assert np.isclose(analytic_overlap(landscape, 0, 1), quadrature, rtol=1e-5, atol=1e-8)


def test_window_readout_is_symmetric_and_uses_one_sample_tensor() -> None:
    landscape = _landscape()
    system = ActiveWindowSystem(landscape, raw_metric=np.array([[1.0]]))
    samples = np.array([[[-2.0], [2.0]], [[-1.8], [2.1]], [[-2.2], [1.9]]])
    readout = system.readout(samples)
    assert np.allclose(readout, readout.T)
    assert np.allclose(np.diag(readout), 1.0)
    assert 0.0 <= readout[0, 1] <= 1.0


def test_gaussian_hermite_joint_reference_has_normalised_lambda_zero_limit() -> None:
    landscape = _landscape()
    system = ActiveWindowSystem(landscape, raw_metric=np.array([[1.0]]), coupling=0.0)
    result = gaussian_hermite_joint_reference(system, np.array([0, 1, 0]), order=8)
    assert abs(result.log_z) < 1e-12
    assert abs(result.free_energy) < 1e-12
    assert result.component_assignments == 8
    assert result.quadrature_points == 512
    assert np.allclose(np.diag(result.readout), 1.0)
    assert np.all((result.readout[np.triu_indices(3, k=1)] > 0.0) & (result.readout[np.triu_indices(3, k=1)] < 1.0))


def test_gaussian_hermite_joint_reference_matches_grid_for_small_window() -> None:
    landscape = GaussianMixtureLandscape(
        logits=np.array([[0.0], [0.0]]),
        means=np.array([[[-0.4]], [[0.7]]]),
        variances=np.array([[[0.5]], [[0.6]]]),
    )
    system = ActiveWindowSystem(landscape, raw_metric=np.array([[1.0]]), coupling=0.6)
    hermite = gaussian_hermite_joint_reference(system, np.array([0, 1]), order=12)
    grid_log_z, grid_readout = grid_reference_1d(system, np.array([0, 1]), np.linspace(-8.0, 8.0, 121))
    assert np.isclose(hermite.log_z, grid_log_z, atol=5e-3)
    assert np.allclose(hermite.readout, grid_readout, atol=5e-3)


def test_gaussian_hermite_center_marginal_is_normalised() -> None:
    landscape = _landscape()
    system = ActiveWindowSystem(landscape, raw_metric=np.array([[1.0]]), coupling=1.0)
    marginal = gaussian_hermite_center_marginal_1d(system, np.array([0, 1, 0]), order=8)
    assert np.isclose(np.sum(marginal.weights), 1.0)
    assert marginal.states.shape == marginal.weights.shape

