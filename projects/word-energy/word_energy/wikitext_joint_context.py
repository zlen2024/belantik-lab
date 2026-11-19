"""Exact six-spin joint score, independent-spoke control and matched CBOW."""
from word_energy.ising.exact_training import _imports
from word_energy.text8_factorial_models import initial as ising_initial
from word_energy.wikitext_matched_vectors import initial as vector_initial
ARMS=('joint','pair_add','vector','joint_shuffled');SEEDS=(17,29,43)

def initial(arm,seed,vocab_size):
    if arm not in ARMS:raise ValueError('unknown model')
    return vector_initial(seed,vocab_size,True) if arm=='vector' else ising_initial('tied-frequency',seed,vocab_size)

def score_windows(raw,windows,offset,arm,coupling_scale=1.):
    jax,jnp,_=_imports();w=jnp.asarray(windows,dtype=jnp.int32);offset=jnp.asarray(offset,dtype=jnp.float64)
    if arm=='vector':
        vec=raw['shared'];return jnp.sum(vec[w[:,0]]*(vec[w[:,1]]+vec[w[:,2]])*.5,axis=1)+offset[w[:,0]]
    if arm not in ('joint','pair_add','joint_shuffled'):raise ValueError('unknown arm')
    states=jnp.array([[-1.,-1.],[1.,-1.],[-1.,1.],[1.,1.]],dtype=jnp.float64);product=states[:,0]*states[:,1]
    b=2.5*jnp.tanh(raw['biases']);k=2*jnp.tanh(raw['internal']);g=1.5*jax.nn.sigmoid(raw['interaction'])*coupling_scale/2
    logits=states@b.T+product[:,None]*k[None,:];q=jax.nn.softmax(logits,axis=0).T
    factors=jnp.expm1(jnp.sum(states[:,None,:]*states[None,:,:]*g,axis=-1))
    center=q[w[:,0]];p1=q[w[:,1]];p2=q[w[:,2]]
    d1=jnp.einsum('ij,bj->bi',factors,p1);d2=jnp.einsum('ij,bj->bi',factors,p2)
    if arm=='pair_add':
        score=jnp.log1p(jnp.sum(center*d1,axis=1))+jnp.log1p(jnp.sum(center*d2,axis=1))
    else:score=jnp.log1p(jnp.sum(center*(d1+d2+d1*d2),axis=1))
    return score+offset[w[:,0]]

def functions(arm,offset):
    if arm not in ARMS:raise ValueError('unknown arm')
    jax,jnp,optax=_imports();offset=jnp.asarray(offset,dtype=jnp.float64)
    def score(params,windows):return score_windows(params,windows,offset,arm)
    def scores(params,windows,noise):
        n=noise.shape[1];neg=jnp.broadcast_to(windows[:,None,:],(windows.shape[0],n,3)).at[:,:,0].set(noise).reshape((-1,3))
        return score(params,windows),score(params,neg).reshape(noise.shape)
    def loss(params,windows,noise):
        p,n=scores(params,windows,noise);return jnp.mean(jax.nn.softplus(1-p[:,None]+n))
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adam(.01))
    @jax.jit
    def step(params,state,windows,noise):
        value,grad=jax.value_and_grad(loss)(params,windows,noise);updates,state=opt.update(grad,state,params)
        return optax.apply_updates(params,updates),state,value
    return jax.jit(score),jax.jit(scores),jax.jit(loss),step,opt
