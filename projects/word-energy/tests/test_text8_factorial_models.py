import unittest
import numpy as np
from word_energy.ising.exact_training import _free_energy_factory,_imports
from word_energy.text8_factorial_models import initial,score_pairs,functions
from word_energy.text8_canonical_eval import canonical_candidate_scores


class FactorialIsingTest(unittest.TestCase):
    def test_tied_and_role_initial_match_enumeration_and_gradients(self):
        jax,jnp,_=_imports(); vocab=6; offset=jnp.zeros(vocab)
        tied=initial('tied',17,vocab); role=initial('role',17,vocab)
        pairs=jnp.array([[0,1],[1,0],[2,3],[4,5]],dtype=jnp.int32)
        _,_,energy,_,_=_free_energy_factory(True)
        reference=lambda x:jax.vmap(lambda p:-energy(x,p))(pairs)
        tied_score=lambda x:score_pairs(x,pairs,offset,role_specific=False,with_frequency=False)
        role_score=lambda x:score_pairs(x,pairs,offset,role_specific=True,with_frequency=False)
        np.testing.assert_allclose(tied_score(tied),reference(tied),atol=2e-14,rtol=0)
        np.testing.assert_allclose(role_score(role),tied_score(tied),atol=2e-14,rtol=0)
        grad=jax.grad(lambda p:jnp.sum(tied_score(p)))(tied)
        ref_grad=jax.grad(lambda p:jnp.sum(reference(p)))(tied)
        role_grad=jax.grad(lambda p:jnp.sum(role_score(p)))(role)
        for key in tied: np.testing.assert_allclose(grad[key],ref_grad[key],atol=1e-12,rtol=0)
        np.testing.assert_allclose(grad['biases'],role_grad['target_biases']+role_grad['context_biases'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(grad['internal'],role_grad['target_internal']+role_grad['context_internal'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(grad['interaction'],role_grad['interaction'],atol=1e-12,rtol=0)

    def test_zero_and_frequency_are_directed_and_exact(self):
        _,jnp,_=_imports(); vocab=5; offset=jnp.asarray(.25*np.log(np.array([1,2,5,11,20],dtype=float)))
        pairs=jnp.array([[0,1],[1,0],[2,4]],dtype=jnp.int32)
        for arm in ('tied','tied-frequency','role','role-frequency'):
            params=initial(arm,17,vocab)
            result=np.asarray(score_pairs(params,pairs,offset,role_specific=arm.startswith('role'),
                                          with_frequency=arm.endswith('frequency'),coupling_scale=0.0))
            expected=np.asarray(offset)[np.asarray(pairs)[:,1]] if arm.endswith('frequency') else np.zeros(3)
            np.testing.assert_array_equal(result,expected)
        self.assertNotEqual(float(offset[1]),float(offset[0]))

    def test_order_and_chunk_do_not_erase_offset(self):
        jax,jnp,_=_imports(); offset=.25*np.log(np.array([1,2,5,11,20],dtype=float))
        params=initial('tied-frequency',17,5); score,_,_,_=functions('tied-frequency',offset)
        pairs=np.array([[0,1],[1,0],[2,3]],dtype=np.int32)
        noise=np.array([[1,2],[0,2],[3,4]],dtype=np.int32)
        p,n=canonical_candidate_scores(params,score,pairs,noise,symmetric=False,chunk_size=4)
        perm=np.array([2,0,1]); p2,n2=canonical_candidate_scores(params,score,pairs[perm],noise[perm],symmetric=False,chunk_size=9)
        np.testing.assert_allclose(p2,p[perm],atol=1e-14,rtol=0)
        np.testing.assert_allclose(n2,n[perm],atol=1e-14,rtol=0)
        self.assertAlmostEqual(float(p[0]-p[1]),float(offset[1]-offset[0]),places=12)
        np.testing.assert_array_equal(n[:,0],p)
        order=np.array([1,0]); p3,n3=canonical_candidate_scores(params,score,pairs,noise[:,order],symmetric=False,chunk_size=4)
        np.testing.assert_array_equal(p3,p); np.testing.assert_array_equal(n3,n[:,order])

    def test_role_specific_16_state_reference_and_gradient(self):
        jax,jnp,_=_imports(); offset=jnp.zeros(4); raw=initial('role',29,4)
        raw['context_biases']=raw['context_biases'].at[1,0].add(.3)
        raw['context_internal']=raw['context_internal'].at[2].add(-.2)
        pairs=jnp.array([[0,1],[2,3]],dtype=jnp.int32)
        states=jnp.array([[-1.,-1.],[1.,-1.],[-1.,1.],[1.,1.]])
        def manual(params,pair):
            tb=2.5*jnp.tanh(params['target_biases'][pair[0]])
            cb=2.5*jnp.tanh(params['context_biases'][pair[1]])
            tk=2*jnp.tanh(params['target_internal'][pair[0]])
            ck=2*jnp.tanh(params['context_internal'][pair[1]])
            g=1.5*jax.nn.sigmoid(params['interaction'])
            x=states@tb+tk*states[:,0]*states[:,1]
            y=states@cb+ck*states[:,0]*states[:,1]
            coupling=jnp.sum(states[:,None,:]*states[None,:,:]*g,axis=-1)
            return jax.scipy.special.logsumexp(x[:,None]+y[None,:]+coupling)-jax.scipy.special.logsumexp(x)-jax.scipy.special.logsumexp(y)
        reference=lambda p:jax.vmap(lambda row:manual(p,row))(pairs)
        actual=lambda p:score_pairs(p,pairs,offset,role_specific=True,with_frequency=False)
        np.testing.assert_allclose(actual(raw),reference(raw),atol=2e-14,rtol=0)
        a=jax.grad(lambda p:jnp.sum(actual(p)))(raw); b=jax.grad(lambda p:jnp.sum(reference(p)))(raw)
        for key in raw: np.testing.assert_allclose(a[key],b[key],atol=1e-12,rtol=0)


if __name__=='__main__': unittest.main()
