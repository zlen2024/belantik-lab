import numpy as np

from word_energy.ising import IsingWordEnergy


def _fixture(coupling_scale: float = 1.0) -> IsingWordEnergy:
    return IsingWordEnergy(
        biases=np.array([[0.0, 0.0], [2.0, 2.0], [-2.0, -2.0]]),
        internal_couplings=np.array([1.0, 0.0, 0.0]),
        interaction_weights=np.array([0.5, 0.5]),
        coupling_scale=coupling_scale,
    )


def test_hand_set_finance_fixture_prefers_positive_bank_basin() -> None:
    model = _fixture()
    assert np.isclose(model.bank_region_mass(np.array([0, 1, 1]), True), 0.9421, atol=2e-4)
    assert np.isclose(model.bank_region_mass(np.array([0, 1, 1]), False), 0.0204, atol=2e-4)


def test_river_context_mirrors_finance_context() -> None:
    model = _fixture()
    finance = model.bank_region_mass(np.array([0, 1, 1]), True)
    river = model.bank_region_mass(np.array([0, 2, 2]), False)
    assert np.isclose(finance, river, atol=1e-12)


def test_normalised_free_energy_is_zero_without_interactions() -> None:
    model = _fixture(coupling_scale=0.0)
    assert abs(model.normalised_free_energy(np.array([0, 1, 1]))) < 1e-12


def test_joint_readout_is_symmetric() -> None:
    readout = _fixture().joint_readout(np.array([0, 1, 2]))
    assert np.allclose(readout, readout.T)
    assert np.allclose(np.diag(readout), 1.0)

