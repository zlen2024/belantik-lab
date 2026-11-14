"""Known-answer checks for the weighted-mean intervention."""
import unittest
from pathlib import Path
import numpy as np

from scripts.run_ising_family_mean_transfer import load_initial, transfer
from word_energy.ising.exact_training import prepare_data


class FamilyMeanTransferTest(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).resolve().parents[1] / 'experiments/runs/ising-attribution-original-seed29.json').exists() and (Path(__file__).resolve().parents[1] / 'experiments/runs/ising-attribution-original-seed43.json').exists(), 'local replay artifacts required')
    def test_reciprocal_transfer_preserves_residuals_and_unselected_parameters(self):
        _, a = load_initial(29)
        _, b = load_initial(43)
        data = prepare_data()
        for background, donor in ((a,b),(b,a)):
            altered, report = transfer(background, donor, data)
            self.assertLessEqual(report['F']['mean_error'], 1e-12)
            self.assertLessEqual(report['R']['mean_error'], 1e-12)
            self.assertLessEqual(report['F']['residual_error'], 1e-12)
            self.assertLessEqual(report['R']['residual_error'], 1e-12)
            np.testing.assert_array_equal(altered['internal'], background['internal'])
            np.testing.assert_array_equal(altered['interaction'], background['interaction'])
            np.testing.assert_array_equal(altered['biases'][:2], background['biases'][:2])

if __name__ == '__main__':
    unittest.main()
