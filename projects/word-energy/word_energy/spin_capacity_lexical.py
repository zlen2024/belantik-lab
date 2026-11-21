"""Exact pair-energy scores for matched lexical capacity experiments."""
from itertools import combinations
from word_energy.ising.exact_training import _imports

def validate_pair_ids(pairs,vocab_size):
    """Check arrays on host before JIT; JAX gathers can silently clip IDs."""
    import numpy as np
    x=np.asarray(pairs)
    if x.ndim!=2 or x.shape[1]!=2 or not len(x) or x.dtype.kind not in 'iu' or np.any(x<0) or np.any(x>=vocab_size):
        raise ValueError('pairs must contain valid integer IDs and have shape (N, 2)')

def states_for(d,jnp):
    i=jnp.arange(1<<d,dtype=jnp.int32)
    return (2*((i[:,None]>>jnp.arange(d))&1)-1).astype(jnp.float64)

def initial(arm,seed,vocab_size):
    jax,jnp,_=_imports();keys=jax.random.split(jax.random.key(seed),3)
    if arm.startswith('vector'):
        d=int(arm[6:]);return {'shared':.1*jax.random.normal(keys[1],(vocab_size,d),dtype=jnp.float64)}
    if arm not in ('ising4','ising6'):raise ValueError('invalid arm')
    d=int(arm[5:]);return {'biases':.1*jax.random.normal(keys[1],(vocab_size,d),dtype=jnp.float64),
      'internal':.1*jax.random.normal(keys[2],(vocab_size,d*(d-1)//2),dtype=jnp.float64),
      'interaction':jnp.full((d,),jnp.log(.8/(1.5-.8)),dtype=jnp.float64)}

def coupling_values(raw,d,mode='bounded'):
    """Actual shared W; fixed-total mode trains relative coordinates only."""
    jax,jnp,_=_imports()
    if mode=='bounded':return 1.5*jax.nn.sigmoid(raw)/d
    if mode=='fixed':return jnp.full((d,),.8/d,dtype=jnp.float64)
    if mode=='redistribute':
        z=jnp.tanh(raw)
        return (.8+.3*(z-jnp.mean(z)))/d
    raise ValueError('unknown coupling mode')

def scores(params,pairs,offset,arm,coupling=True,coupling_mode='bounded'):
    jax,jnp,_=_imports();x=jnp.asarray(pairs,dtype=jnp.int32);prior=jnp.asarray(offset)
    if arm.startswith('vector'):
        v=params['shared'];return jnp.sum(v[x[:,0]]*v[x[:,1]],axis=1)+prior[x[:,0]]
    if arm not in ('ising4','ising6'):raise ValueError('invalid arm')
    d=int(arm[5:]);s=states_for(d,jnp);prod=jnp.stack([s[:,i]*s[:,j] for i,j in combinations(range(d),2)],axis=1)
    ids=x.reshape(-1)
    logits=(2.5*jnp.tanh(params['biases'][ids]))@s.T+(2*jnp.tanh(params['internal'][ids]))@prod.T
    q=jax.nn.softmax(logits,axis=1).reshape((len(x),2,1<<d))
    g=coupling_values(params['interaction'],d,coupling_mode) if coupling else jnp.zeros((d,),dtype=jnp.float64)
    kernel=jnp.exp((s*g)@s.T)
    value=jnp.einsum('bi,ij,bj->b',q[:,0,:],kernel,q[:,1,:])
    return jnp.log(value)+prior[x[:,0]]
