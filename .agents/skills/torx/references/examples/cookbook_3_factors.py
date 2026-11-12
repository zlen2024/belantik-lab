"""Cookbook 3: custom factor, DFG with shared params, TiledFactor, ChainFactor.
Adapted from torx/examples/15_intro_to_factors.ipynb, then checked here. Run from OUTSIDE master/."""
import equinox as eqx
import jax, jax.numpy as jnp, numpy as np
from torx import AbstractReferenceFactor, ChainFactor, DFG, Site, TiledFactor

BIT = jax.ShapeDtypeStruct((), jnp.int32)
DRIVE = jax.ShapeDtypeStruct((), jnp.float32)


class CoinFactor(AbstractReferenceFactor):
    """No inputs: a pbit that is 1 with probability sigmoid(bias)."""
    input_ports: dict = eqx.field(static=True)
    output_spec: jax.ShapeDtypeStruct = eqx.field(static=True)

    def __init__(self):
        self.input_ports, self.output_spec = {}, BIT

    def init_params(self, key):
        return {"bias": jnp.array(0.4)}

    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        out = jax.random.bernoulli(key, jax.nn.sigmoid(params["bias"])).astype(jnp.int32)
        return (out, None) if return_aux else out


class ConditionalBit(AbstractReferenceFactor):
    """P(out | drive) = Bernoulli(sigmoid(w * drive + b))."""
    input_ports: dict = eqx.field(static=True)
    output_spec: jax.ShapeDtypeStruct = eqx.field(static=True)

    def __init__(self):
        self.input_ports, self.output_spec = {"drive": DRIVE}, BIT

    def init_params(self, key):
        return {"w": jnp.array(2.0), "b": jnp.array(-1.0)}

    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        g = params["w"] * inputs["drive"] + params["b"]
        out = jax.random.bernoulli(key, jax.nn.sigmoid(g)).astype(jnp.int32)
        return (out, None) if return_aux else out


def to_drive(parent_outputs):          # porting_fn: parent outputs -> factor input dict
    return {"drive": parent_outputs[0].astype(jnp.float32)}


# Site(name, factor, parents, porting_fn, param_key, info_key, site_info)
sites = (Site("coin", CoinFactor(), (), (), "src", None, None),
         Site("out", ConditionalBit(), ("coin",), to_drive, "cond", None, None))
graph = DFG(sites, {}, "out")
params = graph.init_params(jax.random.key(0))
print("DFG params keys:", list(params))                 # one entry per distinct param_key
draws = jax.vmap(lambda k: graph.sample(k, {}, params))(jax.random.split(jax.random.key(1), 40000))
pc = float(jax.nn.sigmoid(params["src"]["bias"]))
w, b = params["cond"]["w"], params["cond"]["b"]
exact = (1 - pc) * float(jax.nn.sigmoid(b)) + pc * float(jax.nn.sigmoid(w + b))
print(f"P(out=1): sampled {float(draws.mean()):.3f}  exact {exact:.3f}")

# Param sharing: two sites with the SAME param_key reuse one parameter slice
sites2 = (Site("coin", CoinFactor(), (), (), "src", None, None),
          Site("o1", ConditionalBit(), ("coin",), to_drive, "cond", None, None),
          Site("o2", ConditionalBit(), ("o1",), to_drive, "cond", None, None))
g2 = DFG(sites2, {}, "o2")
p2 = g2.init_params(jax.random.key(0))
print("shared param_key ->", list(p2), "(2 conditional sites, one 'cond' slice)")


class FlipFactor(AbstractReferenceFactor):
    input_ports: dict = eqx.field(static=True)
    output_spec: jax.ShapeDtypeStruct = eqx.field(static=True)

    def __init__(self):
        self.input_ports, self.output_spec = {"state": BIT}, BIT

    def init_params(self, key):
        return {"w": jnp.array(1.5), "b": jnp.array(0.0)}

    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        g = params["b"] + params["w"] * (2 * inputs["state"] - 1)
        out = jax.random.bernoulli(key, jax.nn.sigmoid(g)).astype(jnp.int32)
        return (out, None) if return_aux else out


flip = FlipFactor()
fp = flip.init_params(jax.random.key(0))
tiled = TiledFactor(flip, n_tiles=8, weight_tied=True)
t_out = tiled.sample(jax.random.key(3), {"state": jnp.zeros(8, jnp.int32)}, fp)
ref = jax.vmap(lambda k: flip.sample(k, {"state": jnp.array(0, jnp.int32)}, fp))(
    jax.random.split(jax.random.key(3), 8))
print("TiledFactor == vmap over split keys:", bool(jnp.all(t_out == ref)))
chain = ChainFactor(flip, n_steps=5, feedback_porting_fn="state", weight_tied=True)
c_out = chain.sample(jax.random.key(4), {"state": jnp.array(0, jnp.int32)}, fp)
s = jnp.array(0, jnp.int32)
for k in jax.random.split(jax.random.key(4), 5):
    s = flip.sample(k, {"state": s}, fp)
print("ChainFactor == manual feedback loop:", int(c_out) == int(s))

pc_ = float(jax.nn.sigmoid(0.4)); a_ = float(jax.nn.sigmoid(-1.0)); b_ = float(jax.nn.sigmoid(1.0))
print(f"hand: P(coin) {pc_:.4f}, sigmoid(-1) {a_:.4f}, sigmoid(1) {b_:.4f}, 1-P(coin) {1 - pc_:.4f}, "
      f"terms {(1 - pc_) * a_:.4f} + {pc_ * b_:.4f} = {(1 - pc_) * a_ + pc_ * b_:.4f}")
