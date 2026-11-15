"""Independent known-answer checks of legacy and plan-conformant coupling."""
import unittest
import numpy as np

from word_energy.ising.exact_training import _free_energy_factory, _parameters
from word_energy.ising.model import IsingWordEnergy


class SpokeNormalizationTests(unittest.TestCase):
    def test_golden_energy_f2_f3_and_legacy(self):
        values = dict(biases=np.zeros((3,2)), internal_couplings=np.zeros(3), interaction_weights=np.ones(2))
        fixed = IsingWordEnergy(**values, normalize_spokes=True)
        old = IsingWordEnergy(**values)
        self.assertAlmostEqual(float(fixed.physical_energy(np.arange(2),np.ones((2,2)))), -2)
        self.assertAlmostEqual(float(fixed.physical_energy(np.arange(3),np.ones((3,2)))), -2)
        self.assertAlmostEqual(float(old.physical_energy(np.arange(3),np.ones((3,2)))), -4)

    def test_exact_free_energy_and_center_marginal_repeated_word(self):
        jax,jnp,free,_,center = _free_energy_factory(True)
        raw={'biases':jnp.array([[.12,-.31],[.24,.17]]),'internal':jnp.array([.18,-.28]),'interaction':jnp.array([.1,-.4])}
        par=_parameters(raw,jnp)
        reference=IsingWordEnergy(np.asarray(par['biases']),np.asarray(par['internal']),np.asarray(par['interaction']),normalize_spokes=True)
        tokens=np.array([0,1,1]); states,prob=reference.exact_distribution(tokens)
        biases=np.asarray(par['biases']); internal=np.asarray(par['internal']); weights=np.asarray(par['interaction'])
        independent=np.array([-sum(np.dot(biases[t],s[i])+internal[t]*s[i,0]*s[i,1] for i,t in enumerate(tokens))-.5*sum(np.dot(weights,s[0]*s[i]) for i in (1,2)) for s in states])
        np.testing.assert_allclose(reference.physical_energy(tokens,states),independent,atol=1e-12)
        p=np.exp(-independent-np.max(-independent)); p/=p.sum()
        np.testing.assert_allclose(prob,p,atol=1e-12)
        for a,b in ((0,1),(0,2),(1,2)):
            expected_correlation=np.dot(p,np.mean(states[:,a,:]*states[:,b,:],axis=1))
            self.assertAlmostEqual(reference.joint_readout(tokens)[a,b],expected_correlation,places=12)
        self.assertAlmostEqual(float(free(raw,jnp.asarray(tokens))),reference.normalised_free_energy(tokens),places=12)
        state_order=np.array([[-1,-1],[1,-1],[-1,1],[1,1]])
        expected=np.array([prob[np.all(states[:,0,:]==s,axis=1)].sum() for s in state_order])
        np.testing.assert_allclose(np.asarray(center(raw,jnp.asarray(tokens))),expected,atol=1e-12)
        self.assertAlmostEqual(float(free(raw,jnp.asarray(tokens),0.0)),0.0,places=12)
        pair=jnp.asarray([0,1]); old=_free_energy_factory(False)[2]
        self.assertAlmostEqual(float(free(raw,pair)),float(old(raw,pair)),places=12)
        self.assertAlmostEqual(float(free(raw,jnp.asarray(tokens),0.5)),float(old(raw,jnp.asarray(tokens),0.25)),places=12)
        gradient=jax.grad(lambda p:free(p,jnp.asarray(tokens)))(raw)
        for name in raw:
            for index in np.ndindex(np.asarray(raw[name]).shape):
                h=1e-5; plus={k:np.array(v) for k,v in raw.items()}; minus={k:np.array(v) for k,v in raw.items()}
                plus[name][index]+=h; minus[name][index]-=h
                fd=(float(free(plus,jnp.asarray(tokens)))-float(free(minus,jnp.asarray(tokens))))/(2*h)
                self.assertAlmostEqual(float(gradient[name][index]),fd,delta=1e-6)

if __name__=='__main__':
    unittest.main()
