import numpy as np
from word_energy.shared_multi_context import shared_multi_scores,vector_multi_scores
from word_energy.contextual_substitution import ising_shared_correlations
from word_energy.ising.exact_training import _imports

def test_exact_parity_and_dummy_context():
    rng=np.random.default_rng(119);p={'biases':rng.normal(0,.2,(6,4)), 'internal':rng.normal(0,.2,(6,6)), 'interaction':rng.normal(0,.2,4)}
    rows=np.array([[0,1,2,6,6,3,4],[3,3,3,3,3,4,5]],dtype=np.int32)
    ext=dict(p,biases=np.vstack((p['biases'],np.zeros((1,4)))),internal=np.vstack((p['internal'],np.zeros((1,6)))))
    ref=np.array([ising_shared_correlations(ext,int(w[0]),w[1:5].tolist(),w[5:7]) for w in rows])
    np.testing.assert_allclose(np.asarray(shared_multi_scores(p,rows)),ref,atol=1e-12,rtol=0)
    jax,jnp,_=_imports();params={k:jnp.asarray(v) for k,v in p.items()}
    g=jax.grad(lambda x:jnp.sum(shared_multi_scores(x,rows)))(params)
    assert all(np.isfinite(np.asarray(x)).all() for x in g.values())

def test_vector_padding_and_permutation():
    p={'shared':np.arange(30,dtype=float).reshape(6,5)/30}
    rows=np.array([[0,1,6,6,6,2,3],[0,6,1,6,6,2,3]],dtype=np.int32)
    v=np.asarray(vector_multi_scores(p,rows))
    np.testing.assert_allclose(v[0],v[1],atol=1e-12,rtol=0)
