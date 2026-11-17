"""Classical exact Ising F2 and two-dimensional lexical vector baselines."""
from __future__ import annotations
import numpy as np
from word_energy.ising.exact_training import _free_energy_factory, _imports

ARMS=('ising-ranking','vector-ranking','sgns','ising-shuffled')
SEEDS=(17,29,43)


def initial_parameters(arm,seed,vocab_size):
    jax,jnp,_=_imports()
    if arm not in ARMS: raise ValueError('unknown arm')
    key=jax.random.key(seed); _,a,b=jax.random.split(key,3)
    if arm.startswith('ising'):
        return {'biases':jax.random.normal(a,(vocab_size,2),dtype=jnp.float64)*.1,
                'internal':jax.random.normal(b,(vocab_size,),dtype=jnp.float64)*.1,
                'interaction':jnp.full((2,),jnp.log(.8/(1.5-.8)),dtype=jnp.float64)}
    return {'input':jax.random.normal(a,(vocab_size,2),dtype=jnp.float64)*.1,
            'output':jax.random.normal(b,(vocab_size,2),dtype=jnp.float64)*.1}


def functions(arm):
    jax,jnp,optax=_imports()
    if arm not in ARMS: raise ValueError('unknown arm')
    if arm.startswith('ising'):
        _,_,energy,_,_=_free_energy_factory(True)
        def score(params,pairs):
            return jax.vmap(lambda pair:-energy(params,pair))(pairs)
    else:
        def score(params,pairs):
            return jnp.sum(params['input'][pairs[:,0]]*params['output'][pairs[:,1]],axis=-1)
    def scores(params,pairs,noise):
        p=score(params,pairs)
        left=jnp.broadcast_to(pairs[:,0,None],noise.shape)
        negatives=jnp.stack((left,noise),axis=-1).reshape((-1,2))
        n=score(params,negatives).reshape(noise.shape)
        return p,n
    def loss(params,pairs,noise):
        p,n=scores(params,pairs,noise)
        if arm=='sgns':
            return jnp.mean(jax.nn.softplus(-p)+jax.nn.softplus(n).sum(axis=1))
        return jnp.mean(jax.nn.softplus(1-p[:,None]+n))
    optimizer=optax.chain(optax.clip_by_global_norm(5.0),optax.adam(.01))
    @jax.jit
    def step(params,state,pairs,noise):
        value,gradient=jax.value_and_grad(loss)(params,pairs,noise)
        updates,state=optimizer.update(gradient,state,params)
        return optax.apply_updates(params,updates),state,value
    return jax.jit(scores),jax.jit(loss),step,optimizer


def pair_metrics(positive,negative,center_counts):
    """Noise discrimination; MRR@21 uses expected reciprocal rank over ties."""
    p=np.asarray(positive,dtype=float); n=np.asarray(negative,dtype=float)
    if p.ndim!=1 or n.ndim!=2 or n.shape[0]!=len(p): raise ValueError('score shape mismatch')
    if not np.isfinite(p).all() or not np.isfinite(n).all(): raise FloatingPointError('nonfinite score')
    comparisons=(p[:,None]>n)+.5*(p[:,None]==n)
    per_pair=comparisons.mean(axis=1)
    greater=(n>p[:,None]).sum(axis=1); ties=(n==p[:,None]).sum(axis=1)
    reciprocal=np.zeros(len(p),dtype=float)
    for offset in range(n.shape[1]+1):
        eligible=offset<=ties
        reciprocal[eligible]+=1/(greater[eligible]+offset+1)/(ties[eligible]+1)
    order=np.argsort(np.asarray(center_counts),kind='stable')
    strata=[float(per_pair[part].mean()) if len(part) else None for part in np.array_split(order,3)]
    return {'noise_pairwise_auc':float(per_pair.mean()),'noise_mrr_at_21':float(reciprocal.mean()),
            'auc_by_center_frequency_tercile':strata,'positive_score_mean':float(p.mean()),
            'negative_score_mean':float(n.mean()),'positive_score_std':float(p.std()),
            'negative_score_std':float(n.std()),'tie_fraction':float(np.mean(p[:,None]==n))}
