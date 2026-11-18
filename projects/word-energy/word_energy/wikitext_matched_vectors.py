"""3D vectors with 3V or 6V parameters and identical candidate prior."""
from word_energy.ising.exact_training import _imports

def initial(seed,vocab_size,shared):
    jax,jnp,_=_imports();_,a,b=jax.random.split(jax.random.key(seed),3)
    left=jax.random.normal(a,(vocab_size,3),dtype=jnp.float64)*.1
    return {'shared':left} if shared else {'input':left,'output':jax.random.normal(b,(vocab_size,3),dtype=jnp.float64)*.1}

def functions(offset,shared):
    jax,jnp,optax=_imports();offset=jnp.asarray(offset,dtype=jnp.float64)
    def score(p,pairs):
        left=p['shared'] if shared else p['input'];right=p['shared'] if shared else p['output']
        return jnp.sum(left[pairs[:,0]]*right[pairs[:,1]],axis=-1)+offset[pairs[:,1]]
    def scores(p,pairs,noise):
        pos=score(p,pairs);left=jnp.broadcast_to(pairs[:,0,None],noise.shape)
        neg=jnp.stack((left,noise),axis=-1).reshape((-1,2))
        return pos,score(p,neg).reshape(noise.shape)
    def loss(p,pairs,noise):
        pos,neg=scores(p,pairs,noise);return jnp.mean(jax.nn.softplus(1-pos[:,None]+neg))
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adam(.01))
    @jax.jit
    def step(p,state,pairs,noise):
        value,grad=jax.value_and_grad(loss)(p,pairs,noise);updates,state=opt.update(grad,state,p)
        return optax.apply_updates(p,updates),state,value
    return jax.jit(scores),jax.jit(loss),step,opt
