"""Verify prior vector scoring, gradients, ties and zero-model baseline."""
import unittest
import numpy as np
from word_energy.ising.exact_training import _imports
from word_energy.text8_prior_vector import functions
from word_energy.text8_canonical_eval import canonical_candidate_scores

class PriorVectorTest(unittest.TestCase):
    def test_scores_gradients_and_zero(self):
        jax,jnp,_=_imports(); offset=np.array([.2,.7,-.3]); fn,loss,_,_=functions(offset)
        p={'input':jnp.array([[1.,2.],[3.,-1.],[.5,.1]]),'output':jnp.array([[.2,.4],[-.5,.3],[1.,-.2]])}
        pairs=np.array([[0,1],[1,2]],dtype=np.int32); noise=np.array([[2,1],[0,2]],dtype=np.int32)
        pos,neg=fn(p,pairs,noise); pos=np.asarray(pos);neg=np.asarray(neg)
        expected=np.sum(np.asarray(p['input'])[pairs[:,0]]*np.asarray(p['output'])[pairs[:,1]],axis=1)+offset[pairs[:,1]]
        np.testing.assert_allclose(pos,expected,rtol=0,atol=1e-15)
        raw_input=np.asarray(p['input']);raw_output=np.asarray(p['output'])
        expected_neg=np.array([[raw_input[u]@raw_output[v]+offset[v] for v in row] for u,row in zip(pairs[:,0],noise)])
        np.testing.assert_allclose(neg,expected_neg,rtol=0,atol=1e-15)
        expected_loss=np.logaddexp(0,1-expected[:,None]+expected_neg).mean()
        self.assertAlmostEqual(float(loss(p,pairs,noise)),float(expected_loss),places=14)
        grad=jax.grad(loss)(p,pairs,noise); eps=1e-5
        plus={k:np.array(v) for k,v in p.items()};minus={k:v.copy() for k,v in plus.items()};plus['input'][0,0]+=eps;minus['input'][0,0]-=eps
        numeric=(float(loss(plus,pairs,noise))-float(loss(minus,pairs,noise)))/(2*eps)
        self.assertAlmostEqual(float(grad['input'][0,0]),numeric,places=8)
        cp,cn=canonical_candidate_scores(p,fn,pairs,noise,symmetric=False,chunk_size=2)
        self.assertEqual(cp[0],cn[0,1]);self.assertEqual(cp[1],cn[1,1])
        zeros={k:jnp.zeros_like(v) for k,v in p.items()};z,zn=fn(zeros,pairs,noise)
        np.testing.assert_array_equal(np.asarray(z),offset[pairs[:,1]])
        np.testing.assert_array_equal(np.asarray(zn),offset[noise])

if __name__=='__main__':unittest.main()
