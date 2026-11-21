import itertools
import numpy as np
from scipy.special import expit,softmax
from word_energy.ising.exact_training import _imports
from word_energy.spin_capacity_lexical import initial,scores,states_for,coupling_values,validate_pair_ids

def test_pair_exact_enumeration_symmetric_repeated_and_zero():
    _,jnp,_=_imports();p=initial('ising4',17,4);p={k:np.asarray(v) for k,v in p.items()}
    pairs=np.array([[0,1],[1,0],[0,0]]);prior=np.zeros(4)
    x=np.asarray(scores(p,pairs,prior,'ising4'));states=np.asarray(states_for(4,jnp));comb=list(itertools.combinations(range(4),2))
    products=np.stack([states[:,i]*states[:,j] for i,j in comb],axis=1)
    q=softmax((2.5*np.tanh(p['biases']))@states.T+(2*np.tanh(p['internal']))@products.T,axis=1)
    g=1.5*expit(p['interaction'])/4
    want=np.array([np.log(sum(q[a,i]*q[b,j]*np.exp(np.dot(g,states[i]*states[j])) for i,j in itertools.product(range(16),repeat=2))) for a,b in pairs])
    np.testing.assert_allclose(x,want,atol=1e-12,rtol=0)
    np.testing.assert_allclose(x[0],x[1],atol=1e-12,rtol=0)
    np.testing.assert_allclose(np.asarray(scores(p,pairs,prior,'ising4',coupling=False)),0,atol=1e-12,rtol=0)

def test_six_spin_grad_finite_difference():
    jax,jnp,_=_imports();p=initial('ising6',29,4);pairs=jnp.array([[0,1],[2,3],[0,0]]);prior=jnp.zeros(4)
    f=lambda z:jnp.sum(scores(z,pairs,prior,'ising6'))
    grad=jax.grad(f)(p)
    for key,index in [('biases',(0,0)),('internal',(0,0)),('interaction',(0,))]:
        plus,minus=dict(p),dict(p);plus[key]=p[key].at[index].add(1e-5);minus[key]=p[key].at[index].add(-1e-5)
        num=(float(f(plus))-float(f(minus)))/2e-5
        np.testing.assert_allclose(float(grad[key][index]),num,atol=1e-7,rtol=1e-4)

def test_fixed_total_coupling_bounds_and_trainable_relative_weights():
    jax,jnp,_=_imports()
    for d in (4,6):
        raw=jnp.array(np.linspace(-100,100,d));g=np.asarray(coupling_values(raw,d,'redistribute'))
        np.testing.assert_allclose(g.sum(),.8,atol=1e-14,rtol=0)
        assert np.all(g>0) and np.all(g<1.5/d)
        grad=jax.grad(lambda x:coupling_values(x,d,'redistribute')[0])(jnp.zeros(d))
        assert float(jnp.linalg.norm(grad))>0
        fixed_grad=jax.grad(lambda x:jnp.sum(coupling_values(x,d,'fixed')))(jnp.zeros(d))
        np.testing.assert_allclose(fixed_grad,0,atol=0,rtol=0)

def test_host_rejects_invalid_pair_ids():
    import pytest
    validate_pair_ids(np.array([[0,1],[1,1]]),2)
    for x in (np.array([[0,-1]]),np.array([[0,2]]),np.array([[0.,1.]]),np.array([[0,1,2]])):
        with pytest.raises(ValueError):validate_pair_ids(x,2)

def test_relative_w_gradient_finite_difference():
    jax,jnp,_=_imports();p=initial('ising6',17,4);pairs=jnp.array([[0,1],[2,3],[0,0]]);prior=jnp.zeros(4)
    f=lambda z:jnp.sum(scores(z,pairs,prior,'ising6',coupling_mode='redistribute'))
    grad=jax.grad(f)(p);plus,minus=dict(p),dict(p)
    plus['interaction']=p['interaction'].at[0].add(1e-5);minus['interaction']=p['interaction'].at[0].add(-1e-5)
    num=(float(f(plus))-float(f(minus)))/2e-5
    np.testing.assert_allclose(float(grad['interaction'][0]),num,atol=1e-7,rtol=1e-4)
