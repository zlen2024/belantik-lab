"""Exact shared Ising star over four masked context occurrences and two candidates."""
from word_energy.ising.exact_training import _imports
from word_energy.wikitext_four_spin import state_table,unary_probabilities

def shared_multi_scores(params,windows):
    """Rows: target, four context IDs, candidate A/B; final ID denotes uniform dummy."""
    jax,jnp,_=_imports();w=jnp.asarray(windows,dtype=jnp.int32);states=state_table(jnp)
    q=unary_probabilities(params);q=jnp.concatenate((q,jnp.full((1,16),1/16.)),axis=0)
    g=1.5*jax.nn.sigmoid(params['interaction'])/6
    kernel=jnp.exp((states*g)@states.T);msg=q@kernel.T
    corr=q@(kernel*((states@states.T)/4)).T
    center=w[:,0];a=w[:,5];b=w[:,6]
    base=q[center]*jnp.prod(msg[w[:,1:5]],axis=1)*msg[a]*msg[b]
    denominator=jnp.sum(base,axis=1)
    def score(candidate):return jnp.sum(base*corr[candidate]/msg[candidate],axis=1)/denominator
    return jnp.stack((score(a),score(b)),axis=1)

def vector_multi_scores(params,windows):
    _,jnp,_=_imports();w=jnp.asarray(windows,dtype=jnp.int32);v=params['shared'];n=v.shape[0]
    extended=jnp.concatenate((v,jnp.zeros((1,v.shape[1]))),axis=0)
    c=w[:,1:5];mask=(c!=n).astype(v.dtype)
    context=jnp.sum(extended[c]*mask[:,:,None],axis=1)/jnp.maximum(jnp.sum(mask,axis=1,keepdims=True),1.)
    t=v[w[:,0]];ref=t+context+t*context;a=v[w[:,5]];b=v[w[:,6]]
    return jnp.stack((jnp.sum(ref*a,axis=1),jnp.sum(ref*b,axis=1)),axis=1)
