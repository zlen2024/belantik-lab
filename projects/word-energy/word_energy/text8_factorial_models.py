"""Four classical Ising F2 diagnostic arms; frequency offset is a host readout."""
from __future__ import annotations
from word_energy.ising.exact_training import _imports
from word_energy.text8_lexical_models import initial_parameters

ARMS=('tied','tied-frequency','role','role-frequency')
SEEDS=(17,29,43)


def initial(arm,seed,vocabulary_size):
    if arm not in ARMS: raise ValueError('unknown factorial arm')
    base=initial_parameters('ising-ranking',seed,vocabulary_size)
    if not arm.startswith('role'): return base
    return {'target_biases':base['biases'].copy(),'target_internal':base['internal'].copy(),
            'context_biases':base['biases'].copy(),'context_internal':base['internal'].copy(),
            'interaction':base['interaction'].copy()}


def score_pairs(params,pairs,offset,*,role_specific,with_frequency,coupling_scale=1.0):
    """Stable normalized log overlap; offset uses original candidate word ID."""
    jax,jnp,_=_imports()
    pairs=jnp.asarray(pairs,dtype=jnp.int32)
    states=jnp.array([[-1.,-1.],[1.,-1.],[-1.,1.],[1.,1.]],dtype=jnp.float64)
    product=states[:,0]*states[:,1]
    def probabilities(bias,internal):
        scores=states@(2.5*jnp.tanh(bias)).T+product[:,None]*(2.0*jnp.tanh(internal))[None,:]
        return jax.nn.softmax(scores,axis=0).T
    if role_specific:
        left=probabilities(params['target_biases'],params['target_internal'])
        right=probabilities(params['context_biases'],params['context_internal'])
    else:
        left=probabilities(params['biases'],params['internal']); right=left
    g=1.5*jax.nn.sigmoid(params['interaction'])*coupling_scale
    products=states[:,None,:]*states[None,:,:]
    perturbation=jnp.expm1(jnp.sum(products*g,axis=-1))
    result=jnp.log1p(jnp.einsum('bi,ij,bj->b',left[pairs[:,0]],perturbation,right[pairs[:,1]]))
    return result+offset[pairs[:,1]] if with_frequency else result


def functions(arm,offset):
    if arm not in ARMS: raise ValueError('unknown factorial arm')
    jax,jnp,optax=_imports(); offset=jnp.asarray(offset,dtype=jnp.float64)
    role=arm.startswith('role'); frequency=arm.endswith('frequency')
    def scores(params,pairs,noise):
        positive=score_pairs(params,pairs,offset,role_specific=role,with_frequency=frequency)
        left=jnp.broadcast_to(pairs[:,0,None],noise.shape)
        negative_pairs=jnp.stack((left,noise),axis=-1).reshape((-1,2))
        negative=score_pairs(params,negative_pairs,offset,role_specific=role,with_frequency=frequency).reshape(noise.shape)
        return positive,negative
    def loss(params,pairs,noise):
        p,n=scores(params,pairs,noise)
        return jnp.mean(jax.nn.softplus(1-p[:,None]+n))
    optimizer=optax.chain(optax.clip_by_global_norm(5.0),optax.adam(.01))
    @jax.jit
    def step(params,state,pairs,noise):
        value,gradient=jax.value_and_grad(loss)(params,pairs,noise)
        updates,state=optimizer.update(gradient,state,params)
        return optax.apply_updates(params,updates),state,value
    return jax.jit(scores),jax.jit(loss),step,optimizer
