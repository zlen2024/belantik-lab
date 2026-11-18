"""Fixed-prior vector comparator matching the Ising ranking objective."""
from word_energy.ising.exact_training import _imports
from word_energy.text8_lexical_models import initial_parameters


def initial(seed,vocab_size):
    return initial_parameters('vector-ranking',seed,vocab_size)


def functions(offset):
    jax,jnp,optax=_imports(); prior=jnp.asarray(offset,dtype=jnp.float64)
    def score(params,pairs):
        return jnp.sum(params['input'][pairs[:,0]]*params['output'][pairs[:,1]],axis=-1)+prior[pairs[:,1]]
    def scores(params,pairs,noise):
        pos=score(params,pairs)
        left=jnp.broadcast_to(pairs[:,0,None],noise.shape)
        neg=jnp.stack((left,noise),axis=-1).reshape((-1,2))
        return pos,score(params,neg).reshape(noise.shape)
    def loss(params,pairs,noise):
        pos,neg=scores(params,pairs,noise)
        return jnp.mean(jax.nn.softplus(1-pos[:,None]+neg))
    optimizer=optax.chain(optax.clip_by_global_norm(5.0),optax.adam(.01))
    @jax.jit
    def step(params,state,pairs,noise):
        value,gradient=jax.value_and_grad(loss)(params,pairs,noise)
        updates,state=optimizer.update(gradient,state,params)
        return optax.apply_updates(params,updates),state,value
    return jax.jit(scores),jax.jit(loss),step,optimizer
