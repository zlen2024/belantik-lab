import unittest
import numpy as np

from word_energy.ising.exact_training import _free_energy_factory, _imports
from word_energy.text8_lexical_models import functions,initial_parameters,pair_metrics


class Text8ModelTests(unittest.TestCase):
    def test_ising_f2_matches_exact_and_zero_coupling(self):
        jax,jnp,_=_imports(); params=initial_parameters('ising-ranking',17,5)
        score,_,_,_=functions('ising-ranking')
        pairs=jnp.asarray([[0,1],[1,0],[0,2]],dtype=jnp.int32)
        noise=jnp.asarray([[2,3],[2,4],[1,4]],dtype=jnp.int32)
        positive,_=score(params,pairs,noise)
        _,_,energy,_,_=_free_energy_factory(True)
        expected=np.array([-float(energy(params,p)) for p in pairs])
        np.testing.assert_allclose(positive,expected,atol=1e-12,rtol=0)
        self.assertAlmostEqual(float(positive[0]),float(positive[1]),places=12)
        for p in pairs: self.assertAlmostEqual(float(energy(params,p,0.0)),0,places=12)
        identical=initial_parameters('ising-shuffled',17,5)
        for key in params: np.testing.assert_array_equal(params[key],identical[key])

    def test_loss_formulas_and_gradients(self):
        jax,jnp,_=_imports()
        pairs=jnp.asarray([[0,1],[1,2]],dtype=jnp.int32)
        negatives=jnp.asarray([[2,3,0,1,2],[0,1,3,2,1]],dtype=jnp.int32)
        for arm in ('ising-ranking','vector-ranking','sgns'):
            params=initial_parameters(arm,17,5); scores,loss,step,opt=functions(arm)
            p,n=scores(params,pairs,negatives)
            expected=(np.logaddexp(0,-np.asarray(p))+np.logaddexp(0,np.asarray(n)).sum(axis=1)).mean() if arm=='sgns' else np.logaddexp(0,1-np.asarray(p)[:,None]+np.asarray(n)).mean()
            self.assertAlmostEqual(float(loss(params,pairs,negatives)),float(expected),places=12)
            updated,_,value=step(params,opt.init(params),pairs,negatives)
            self.assertTrue(np.isfinite(float(value)))
            self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in updated.values()))

    def test_all_ties_are_chance(self):
        metric=pair_metrics(np.zeros(6),np.zeros((6,20)),np.arange(1,7))
        self.assertEqual(metric['noise_pairwise_auc'],.5)
        self.assertAlmostEqual(metric['noise_mrr_at_21'],sum(1/i for i in range(1,22))/21)


if __name__=='__main__': unittest.main()
