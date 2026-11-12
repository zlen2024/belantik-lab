import numpy as np

from word_energy.energy_embedding import ActiveWindowSystem, GaussianMixtureLandscape, analytic_overlap


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

