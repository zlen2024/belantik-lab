"""Independent explicit 4096-state and 256-state checks for four-spin star."""
import unittest,itertools
import numpy as np
from word_energy.ising.exact_training import _imports
from word_energy.wikitext_four_spin import initial,score_windows,state_table
class FourSpinTest(unittest.TestCase):
 def test_direct_enumeration_and_gradient(self):
  jax,jnp,_=_imports();p=initial('joint4',17,3);w=jnp.array([[0,1,2],[1,1,2]],jnp.int32);offset=jnp.array([.1,.2,.3]);single=state_table(jnp)
  triples=jnp.asarray(list(itertools.product(range(16),repeat=3)));pairs=jnp.asarray(list(itertools.product(range(16),repeat=2)))
  products=jnp.stack([single[:,i]*single[:,j] for i,j in itertools.combinations(range(4),2)],axis=1)
  def ref(raw,row,joint):
   qlog=single@(2.5*jnp.tanh(raw['biases'])).T+products@(2*jnp.tanh(raw['internal'])).T;qlog=jax.nn.log_softmax(qlog,axis=0).T
   g=1.5*jax.nn.sigmoid(raw['interaction'])/2;c,a,b=row
   if joint:
    sc=single[triples];v=qlog[c,triples[:,0]]+qlog[a,triples[:,1]]+qlog[b,triples[:,2]]+jnp.sum(g*sc[:,0]*sc[:,1],axis=1)+jnp.sum(g*sc[:,0]*sc[:,2],axis=1)
    return jax.scipy.special.logsumexp(v)+offset[c]
   sp=single[pairs]
   def one(k):return jax.scipy.special.logsumexp(qlog[c,pairs[:,0]]+qlog[k,pairs[:,1]]+jnp.sum(g*sp[:,0]*sp[:,1],axis=1))
   return one(a)+one(b)+offset[c]
  for arm in ('joint4','pair_add4'):
   f=lambda x:jnp.sum(score_windows(x,w,offset,arm));r=lambda x:sum(ref(x,row,arm=='joint4') for row in w)
   np.testing.assert_allclose(f(p),r(p),atol=3e-14,rtol=0)
   x=jax.grad(f)(p);y=jax.grad(r)(p)
   for k in x:np.testing.assert_allclose(x[k],y[k],atol=1e-11,rtol=1e-10)
   eps=1e-5;a={k:np.array(v) for k,v in p.items()};b={k:v.copy() for k,v in a.items()};a['biases'][0,0]+=eps;b['biases'][0,0]-=eps
   self.assertAlmostEqual(float(x['biases'][0,0]),(float(f(a))-float(f(b)))/(2*eps),places=8)
   np.testing.assert_array_equal(score_windows(p,w,offset,arm,0.),np.asarray(offset)[np.asarray(w)[:,0]])
   np.testing.assert_allclose(score_windows(p,w,offset,arm),score_windows(p,np.asarray(w)[:,[0,2,1]],offset,arm),atol=2e-15,rtol=0)
 def test_two_spin_model_is_approached_by_neutral_extra_spins(self):
  _,jnp,_=_imports()
  from word_energy.wikitext_joint_context import initial as old_initial,score_windows as old_score
  old=old_initial('joint',29,4);new={'biases':jnp.zeros((4,4)),'internal':jnp.zeros((4,6)),'interaction':jnp.full((4,),-60.)}
  new['biases']=new['biases'].at[:,:2].set(old['biases']);new['internal']=new['internal'].at[:,0].set(old['internal']);new['interaction']=new['interaction'].at[:2].set(old['interaction'])
  w=jnp.array([[0,1,2],[1,1,3]],dtype=jnp.int32);o=jnp.array([.1,.2,.3,.4])
  np.testing.assert_allclose(score_windows(new,w,o,'joint4'),old_score(old,w,o,'joint'),atol=2e-14,rtol=0)
  np.testing.assert_allclose(score_windows(new,w,o,'pair_add4'),old_score(old,w,o,'pair_add'),atol=2e-14,rtol=0)
 def test_vector_matched_and_zero(self):
  _,jnp,_=_imports();p=initial('vector10',17,5);self.assertEqual(np.asarray(p['shared']).size,50)
  w=jnp.array([[0,1,2],[2,2,3]]);o=jnp.array([.1,.2,.3,.4,.5]);v=np.asarray(p['shared']);a=np.asarray(w)
  np.testing.assert_allclose(score_windows(p,w,o,'vector10'),np.sum(v[a[:,0]]*(v[a[:,1]]+v[a[:,2]])/2,axis=1)+np.asarray(o)[a[:,0]],atol=1e-15,rtol=0)
  z={'shared':jnp.zeros_like(p['shared'])};np.testing.assert_array_equal(score_windows(z,w,o,'vector10'),np.asarray(o)[a[:,0]])
if __name__=='__main__':unittest.main()
