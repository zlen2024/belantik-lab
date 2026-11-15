"""Meaningful checks for train-only Ising spectral initialization."""
import unittest
from unittest.mock import patch
import numpy as np

from word_energy.ising.exact_training import prepare_data
from word_energy.ising.spectral_initializer import spectral_biases


class SpectralInitializerTests(unittest.TestCase):
    def test_counts_determinism_and_permutation(self):
        data=prepare_data(); tokens=data['train_windows']; size=len(data['ids'])
        a=spectral_biases(tokens,size); b=spectral_biases(tokens.copy(),size)
        np.testing.assert_array_equal(a['counts'],a['counts'].T)
        self.assertEqual(int(a['counts'].sum()),len(tokens)*6)
        np.testing.assert_array_equal(a['biases'],b['biases'])
        np.testing.assert_array_equal(a['permuted_biases'],a['biases'][a['permutation']])
        np.testing.assert_allclose(a['biases'].mean(axis=0),0,atol=1e-14)
        np.testing.assert_allclose(a['biases'].std(axis=0),0.1,atol=1e-14)
        data['validation_windows'][:]=0; data['test_windows'][:]=0
        data['train_families']=tuple('changed' for _ in data['train_families'])
        np.testing.assert_array_equal(a['biases'],spectral_biases(data['train_windows'],size)['biases'])

    def test_rejects_degenerate_and_out_of_range(self):
        with self.assertRaises(ValueError): spectral_biases(np.zeros((4,3),dtype=int),4)
        with self.assertRaises(ValueError): spectral_biases(np.array([[0,1,4]]),4)

    def test_exception_preserves_snapshots_and_stays_failed(self):
        from scripts.run_ising_spectral_ablation import persist_partial_on_error
        captured={'stage1_lexical_20':{'biases':[[0.1,0.2]]}}
        with patch('scripts.run_ising_spectral_ablation.log_contract') as log, patch('scripts.run_ising_spectral_ablation.mlflow.set_tag') as tag, patch('scripts.run_ising_spectral_ablation.sha256_file',return_value='fixture'):
            with self.assertRaisesRegex(RuntimeError,'test failure'):
                with persist_partial_on_error(101,'random',captured):
                    raise RuntimeError('test failure')
            self.assertEqual(log.call_args.args[2]['partial_snapshots'],captured)
            tag.assert_called_once_with('partial_snapshots_captured','true')

if __name__=='__main__':
    unittest.main()
