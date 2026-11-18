"""Check parameter matching, exact scores, offsets, gradients and ID ties."""
import unittest
import numpy as np
from word_energy.ising.exact_training import _imports
from word_energy.wikitext_matched_vectors import initial,functions
from word_energy.text8_canonical_eval import canonical_candidate_scores

class MatchedVectorTest(unittest.TestCase):
 def test_models(self):
  jax,jnp,_=_imports();prior=np.array([.1,.2,.3]);pairs=np.array([[0,1],[1,2]],np.int32);noise=np.array([[1,2],[0,2]],np.int32)
  for shared in (True,False):
   p=initial(17,3,shared);self.assertEqual(sum(np.asarray(v).size for v in p.values()),9 if shared else 18)
   score,loss,_,_=functions(prior,shared);pos,neg=score(p,pairs,noise);left=np.asarray(p['shared'] if shared else p['input']);right=np.asarray(p['shared'] if shared else p['output'])
   np.testing.assert_allclose(pos,np.sum(left[pairs[:,0]]*right[pairs[:,1]],axis=-1)+prior[pairs[:,1]],atol=1e-15,rtol=0)
   cp,cn=canonical_candidate_scores(p,score,pairs,noise,symmetric=False,chunk_size=2);self.assertEqual(cp[0],cn[0,0]);self.assertEqual(cp[1],cn[1,1])
   grad=jax.grad(loss)(p,pairs,noise);key='shared' if shared else 'input';eps=1e-5
   pp={k:np.array(v) for k,v in p.items()};pm={k:v.copy() for k,v in pp.items()};pp[key][0,0]+=eps;pm[key][0,0]-=eps
   self.assertAlmostEqual(float(grad[key][0,0]),(float(loss(pp,pairs,noise))-float(loss(pm,pairs,noise)))/(2*eps),places=8)
   z={k:jnp.zeros_like(v) for k,v in p.items()};p0,n0=score(z,pairs,noise);np.testing.assert_array_equal(p0,prior[pairs[:,1]]);np.testing.assert_array_equal(n0,prior[noise])
if __name__=='__main__':unittest.main()
