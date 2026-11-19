"""Exact normalized four-spin unary and twelve-spin joint-star models."""
from itertools import combinations
from word_energy.ising.exact_training import _imports
ARMS=('joint4','pair_add4','vector10','joint4_shuffled');SEEDS=(17,29,43)
PAIRS=tuple(combinations(range(4),2))

def initial(arm,seed,vocab_size):
    if arm not in ARMS:raise ValueError('unknown model')
    jax,jnp,_=_imports();_,a,b=jax.random.split(jax.random.key(seed),3)
    if arm=='vector10':return {'shared':.1*jax.random.normal(a,(vocab_size,10),dtype=jnp.float64)}
    return {'biases':.1*jax.random.normal(a,(vocab_size,4),dtype=jnp.float64),
            'internal':.1*jax.random.normal(b,(vocab_size,6),dtype=jnp.float64),
            'interaction':jnp.full((4,),jnp.log(.8/(1.5-.8)),dtype=jnp.float64)}

def state_table(jnp):
    index=jnp.arange(16,dtype=jnp.int32)
    return (2*((index[:,None]>>jnp.arange(4))&1)-1).astype(jnp.float64)

def unary_probabilities(raw):
    jax,jnp,_=_imports();states=state_table(jnp);products=jnp.stack([states[:,i]*states[:,j] for i,j in PAIRS],axis=1)
    logits=states@(2.5*jnp.tanh(raw['biases'])).T+products@(2*jnp.tanh(raw['internal'])).T
    return jax.nn.softmax(logits,axis=0).T

def score_windows(raw,windows,offset,arm,coupling_scale=1.):
    jax,jnp,_=_imports();w=jnp.asarray(windows,dtype=jnp.int32);offset=jnp.asarray(offset,dtype=jnp.float64)
    if arm=='vector10':
        v=raw['shared'];return jnp.sum(v[w[:,0]]*(v[w[:,1]]+v[w[:,2]])*.5,axis=1)+offset[w[:,0]]
    if arm not in ('joint4','pair_add4','joint4_shuffled'):raise ValueError('unknown model')
    states=state_table(jnp);q=unary_probabilities(raw);g=1.5*jax.nn.sigmoid(raw['interaction'])*coupling_scale/2
    kernel=jnp.expm1(jnp.sum(states[:,None,:]*states[None,:,:]*g,axis=-1))
    center=q[w[:,0]];d1=jnp.einsum('ij,bj->bi',kernel,q[w[:,1]]);d2=jnp.einsum('ij,bj->bi',kernel,q[w[:,2]])
    if arm=='pair_add4':score=jnp.log1p(jnp.sum(center*d1,axis=1))+jnp.log1p(jnp.sum(center*d2,axis=1))
    else:score=jnp.log1p(jnp.sum(center*(d1+d2+d1*d2),axis=1))
    return score+offset[w[:,0]]

def functions(arm,offset):
    if arm not in ARMS:raise ValueError('unknown model')
    jax,jnp,optax=_imports();offset=jnp.asarray(offset,dtype=jnp.float64)
    def score(p,w):return score_windows(p,w,offset,arm)
    def scores(p,w,noise):
        n=noise.shape[1];neg=jnp.broadcast_to(w[:,None,:],(w.shape[0],n,3)).at[:,:,0].set(noise).reshape((-1,3))
        return score(p,w),score(p,neg).reshape(noise.shape)
    def loss(p,w,noise):
        pos,neg=scores(p,w,noise);return jnp.mean(jax.nn.softplus(1-pos[:,None]+neg))
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adam(.01))
    @jax.jit
    def step(p,state,w,noise):
        value,grad=jax.value_and_grad(loss)(p,w,noise);updates,state=opt.update(grad,state,p)
        return optax.apply_updates(p,updates),state,value
    return jax.jit(score),jax.jit(scores),jax.jit(loss),step,opt
