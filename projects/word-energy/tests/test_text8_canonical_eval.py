import unittest
import numpy as np
from word_energy.text8_canonical_eval import canonical_candidate_scores,evaluate,stable_tied_ising_scores,stable_ising_score_adapter
from word_energy.ising.exact_training import _free_energy_factory,_imports
from word_energy.text8_lexical_models import initial_parameters


class CanonicalEvaluationTest(unittest.TestCase):
    @staticmethod
    def mock_scores(params,pairs,noise):
        # Batch-shape-specific offsets emulate numerical false ties.
        values=np.asarray(pairs)[:,0]*100+np.asarray(pairs)[:,1]
        return values.astype(float),np.zeros_like(noise,dtype=float)

    def test_duplicate_ids_share_exact_values(self):
        pairs=np.array([[2,4],[2,5]],dtype=np.int32)
        noise=np.array([[4,5,4],[5,4,5]],dtype=np.int32)
        p,n=canonical_candidate_scores(None,self.mock_scores,pairs,noise,chunk_size=3)
        self.assertTrue(np.array_equal(n[:,0],p))
        self.assertEqual(n[0,1],p[1]); self.assertEqual(n[1,1],p[0])

    def test_symmetric_pair_and_direction(self):
        pairs=np.array([[2,4],[4,2]],dtype=np.int32); noise=np.array([[4],[2]],dtype=np.int32)
        symmetric,_=canonical_candidate_scores(None,self.mock_scores,pairs,noise,symmetric=True,chunk_size=3)
        directed,_=canonical_candidate_scores(None,self.mock_scores,pairs,noise,symmetric=False,chunk_size=3)
        self.assertEqual(symmetric[0],symmetric[1]); self.assertNotEqual(directed[0],directed[1])

    def test_zero_coupling_all_candidates_tie(self):
        pairs=np.array([[1,3],[2,4],[3,1]],dtype=np.int32)
        noise=np.tile(np.arange(20,dtype=np.int32),(3,1))
        m=evaluate(None,self.mock_scores,pairs,noise,np.ones(3),symmetric=True,zero_coupling=True)
        self.assertEqual(m['noise_pairwise_auc'],.5)
        self.assertAlmostEqual(m['noise_mrr_at_21'],sum(1/i for i in range(1,22))/21)

    def test_stable_ising_values_gradients_and_analytic_zero(self):
        jax,jnp,_=_imports(); params=initial_parameters('ising-ranking',17,5)
        pairs=jnp.array([[0,1],[1,0],[2,3],[4,4]],dtype=jnp.int32)
        _,_,energy,_,_=_free_energy_factory(True)
        reference=lambda r: jax.vmap(lambda pair:-energy(r,pair))(pairs)
        stable=lambda r: stable_tied_ising_scores(r,pairs)
        np.testing.assert_allclose(stable(params),reference(params),atol=2e-14,rtol=0)
        g1=jax.grad(lambda r:jnp.sum(stable(r)))(params)
        g2=jax.grad(lambda r:jnp.sum(reference(r)))(params)
        for key in params: np.testing.assert_allclose(g1[key],g2[key],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(stable_tied_ising_scores(params,pairs,0.0),np.zeros(len(pairs)))

    def test_distinct_ids_identical_landscape_and_chunk_invariance(self):
        jax,jnp,_=_imports(); params=initial_parameters('ising-ranking',17,5)
        params['biases']=params['biases'].at[2].set(params['biases'][1])
        params['internal']=params['internal'].at[2].set(params['internal'][1])
        pairs=np.array([[0,1],[1,0],[0,2]],dtype=np.int32)
        noise=np.array([[2,1],[0,2],[1,2]],dtype=np.int32)
        fn=jax.jit(stable_ising_score_adapter)
        p,n=canonical_candidate_scores(params,fn,pairs,noise,symmetric=True,chunk_size=2)
        p2,n2=canonical_candidate_scores(params,fn,pairs,noise,symmetric=True,chunk_size=7)
        np.testing.assert_array_equal(p,p2); np.testing.assert_array_equal(n,n2)
        self.assertEqual(p[0],p[1]); self.assertEqual(p[0],p[2]); self.assertEqual(p[0],n[0,0])


if __name__=='__main__': unittest.main()
