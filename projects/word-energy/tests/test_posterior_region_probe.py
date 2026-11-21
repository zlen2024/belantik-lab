import itertools
import numpy as np
from scipy.special import expit
from word_energy.posterior_region_probe import posterior_and_regions
from word_energy.wikitext_four_spin_wic import STATES,unary

def test_factorization_matches_full_3_token_enumeration_and_dummy_identity():
    rng=np.random.default_rng(410);p={'biases':rng.normal(0,.2,(5,4)), 'internal':rng.normal(0,.2,(5,6)),'interaction':rng.normal(0,.2,4)}
    t=np.array([0]);ctx=np.array([[1,2,5,5]]);a=np.array([3]);b=np.array([4])
    post,prior,region,qa,qb=posterior_and_regions(p,t,ctx,a,b)
    q=unary(p);g=1.5*expit(p['interaction'])/6;weights=np.zeros(16)
    for i,j,k in itertools.product(range(16),repeat=3):
        weights[i]+=q[0,i]*q[1,j]*q[2,k]*np.exp(np.dot(g,STATES[i]*(STATES[j]+STATES[k])))
    weights/=weights.sum();np.testing.assert_allclose(post[0],weights,atol=1e-12,rtol=0)
    np.testing.assert_allclose(prior[0],q[0],atol=1e-12,rtol=0)
    np.testing.assert_allclose(region[0]+((qb[0]>qa[0])+.5*(qb[0]==qa[0])),1.,atol=0,rtol=0)
    empty,*_=posterior_and_regions(p,t,np.array([[5,5,5,5]]),a,b)
    np.testing.assert_allclose(empty,prior,atol=1e-12,rtol=0)
    reverse,*_=posterior_and_regions(p,t,ctx[:,::-1],a,b)
    np.testing.assert_allclose(reverse,post,atol=1e-12,rtol=0)
