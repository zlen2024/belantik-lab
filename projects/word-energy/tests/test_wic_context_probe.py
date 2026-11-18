"""Check star posterior against direct enumeration, JS and span extraction."""
import unittest
from itertools import product
import numpy as np
from scipy.special import expit
from word_energy.wic_context_probe import STATES,star,unary,negative_js,extract,auc
class WiCTest(unittest.TestCase):
 def test_posterior_direct(self):
  raw={'target_biases':np.array([[.2,-.1],[.4,.3]]),'target_internal':np.array([.1,-.2]),'context_biases':np.array([[-.3,.2],[.2,-.4]]),'context_internal':np.array([.2,.1]),'interaction':np.array([-.1,.3])}
  p=star(raw,0,[1],True);pt=unary(raw['target_biases'][0],raw['target_internal'][0]);q=unary(raw['context_biases'][1],raw['context_internal'][1]);g=1.5*expit(raw['interaction'])
  weights=np.array([pt[i]*q[j]*np.exp((STATES[i]*STATES[j])@g) for i,j in product(range(4),repeat=2)]).reshape(4,4)
  np.testing.assert_allclose(p,weights.sum(axis=1)/weights.sum(),atol=1e-15);self.assertAlmostEqual(negative_js(p,p),0.)
 def test_token_spans_and_auc(self):
  v={'river':1,'money':2};self.assertEqual(extract('a bank by river',2,6,v),[1]);self.assertEqual(extract('bank money',0,4,v),[2]);self.assertIsNone(extract('bank',0,0,v));self.assertEqual(auc([1,0],[.3,.3]),.5)
if __name__=='__main__':unittest.main()
