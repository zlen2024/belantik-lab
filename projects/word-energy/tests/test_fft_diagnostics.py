"""Compare FFT and existing quadratic autocovariance diagnostics."""
import unittest
import numpy as np
from word_energy.energy_embedding.mcmc_diagnostics import split_rank_rhat_ess
from word_energy.ising.fft_diagnostics import split_rank_rhat_ess_fft
class DiagnosticsTest(unittest.TestCase):
 def test_fft_matches_reference(self):
  rng=np.random.default_rng(912)
  for x in (rng.normal(size=(4,241)),rng.integers(0,2,size=(4,401)),np.ones((4,20))):
   a=split_rank_rhat_ess(x);b=split_rank_rhat_ess_fft(x)
   np.testing.assert_allclose(a,b,atol=1e-11,rtol=1e-11)
if __name__=='__main__':unittest.main()
