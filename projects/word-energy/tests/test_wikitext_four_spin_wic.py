"""Context posterior is exact sum over 4096 joint spin states."""
import unittest,itertools
import numpy as np
from scipy.special import expit
from word_energy.wikitext_four_spin import initial
from word_energy.wikitext_four_spin_wic import STATES,unary,star
class WicFourTest(unittest.TestCase):
 def test_4096_state_posterior(self):
  p={k:np.asarray(v) for k,v in initial('joint4',17,3).items()};q=unary(p);g=1.5*expit(p['interaction'])/2
  pairs=np.array(list(itertools.product(range(16),repeat=3)));s=STATES[pairs]
  weights=q[0,pairs[:,0]]*q[1,pairs[:,1]]*q[1,pairs[:,2]]*np.exp(np.sum(g*s[:,0]*s[:,1],axis=1)+np.sum(g*s[:,0]*s[:,2],axis=1))
  actual=np.bincount(pairs[:,0],weights=weights,minlength=16);actual/=actual.sum()
  np.testing.assert_allclose(star(p,0,[1,1]),actual,atol=1e-14,rtol=0)
  np.testing.assert_allclose(star(p,0,[1,2]),star(p,0,[2,1]),atol=1e-15,rtol=0)
if __name__=='__main__':unittest.main()
