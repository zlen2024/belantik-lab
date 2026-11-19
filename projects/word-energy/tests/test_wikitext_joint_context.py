"""Analytic factorization/gradients match explicit 64-state graph."""
import unittest,itertools
import numpy as np
from word_energy.ising.exact_training import _imports
from word_energy.wikitext_joint_context import initial,score_windows

class JointTest(unittest.TestCase):
 def test_joint_and_independent_against_enumeration(self):
  jax,jnp,_=_imports();p=initial('joint',17,4);windows=jnp.array([[0,1,2],[1,1,3]],dtype=jnp.int32);offset=jnp.array([.1,.2,.3,.4])
  states=jnp.asarray(list(itertools.product((-1.,1.),repeat=6))).reshape((-1,3,2));single=jnp.asarray(list(itertools.product((-1.,1.),repeat=2)))
  def reference(raw,window,joint):
   b=2.5*jnp.tanh(raw['biases']);k=2*jnp.tanh(raw['internal']);g=1.5*jax.nn.sigmoid(raw['interaction'])/2
   def U(s,index):return jnp.sum(s*b[index],axis=-1)+k[index]*s[...,0]*s[...,1]
   def Z(index):return jax.scipy.special.logsumexp(U(single,index))
   c,a,d=window
   if joint:
    L=U(states[:,0],c)+U(states[:,1],a)+U(states[:,2],d)+jnp.sum(g*states[:,0]*states[:,1],axis=-1)+jnp.sum(g*states[:,0]*states[:,2],axis=-1)
    return jax.scipy.special.logsumexp(L)-Z(c)-Z(a)-Z(d)+offset[c]
   pair=jnp.asarray(list(itertools.product((-1.,1.),repeat=4))).reshape((-1,2,2))
   def one(other):return jax.scipy.special.logsumexp(U(pair[:,0],c)+U(pair[:,1],other)+jnp.sum(g*pair[:,0]*pair[:,1],axis=-1))-Z(c)-Z(other)
   return one(a)+one(d)+offset[c]
  for arm in ('joint','pair_add'):
   f=lambda p:jnp.sum(score_windows(p,windows,offset,arm))
   ref=lambda p:sum(reference(p,w,arm=='joint') for w in windows)
   np.testing.assert_allclose(f(p),ref(p),rtol=0,atol=2e-14)
   grad=jax.grad(f)(p);expected=jax.grad(ref)(p)
   for k in grad:np.testing.assert_allclose(grad[k],expected[k],rtol=1e-11,atol=1e-12)
   eps=1e-5;plus={k:np.array(v) for k,v in p.items()};minus={k:v.copy() for k,v in plus.items()}
   plus['biases'][0,0]+=eps;minus['biases'][0,0]-=eps
   self.assertAlmostEqual(float(grad['biases'][0,0]),(float(f(plus))-float(f(minus)))/(2*eps),places=8)
   z=score_windows(p,windows,offset,arm,coupling_scale=0.);np.testing.assert_array_equal(np.asarray(z),np.asarray(offset)[np.asarray(windows)[:,0]])
   swapped=np.asarray(windows)[:,[0,2,1]];np.testing.assert_allclose(score_windows(p,windows,offset,arm),score_windows(p,swapped,offset,arm),rtol=0,atol=1e-15)
 def test_vector_matches_explicit(self):
  _,jnp,_=_imports();p=initial('vector',17,4);w=jnp.array([[0,1,2],[1,1,3]],dtype=jnp.int32);o=jnp.array([.1,.2,.3,.4]);v=np.asarray(p['shared']);x=np.asarray(w)
  exp=np.sum(v[x[:,0]]*(v[x[:,1]]+v[x[:,2]])/2,axis=1)+np.asarray(o)[x[:,0]]
  np.testing.assert_allclose(score_windows(p,w,o,'vector'),exp,rtol=0,atol=1e-15)
if __name__=='__main__':unittest.main()
