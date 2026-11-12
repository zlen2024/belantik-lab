# Factors, DFG, composites, hybrid and Gaussian circuits (all snippets verified)

Runnable versions: `examples/cookbook_2_hybrid.py`, `examples/cookbook_3_factors.py`. Public import path is `from torx import ...` for factors and DFG, `from torx import psc` for gates and simulators.

## Factor contract (`torx/factor.py`)

A factor is a conditional `P(output | inputs)` with parameters held **outside** the object.

```python
class ConditionalBit(AbstractReferenceFactor):          # AbstractReferenceFactor gives sample_with_references for free
    input_ports: dict = eqx.field(static=True)          # {"drive": jax.ShapeDtypeStruct((), jnp.float32)}
    output_spec: jax.ShapeDtypeStruct = eqx.field(static=True)
    def __init__(self): self.input_ports, self.output_spec = {"drive": DRIVE}, BIT
    def init_params(self, key): return {"w": jnp.array(2.0), "b": jnp.array(-1.0)}
    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        out = jax.random.bernoulli(key, jax.nn.sigmoid(params["w"]*inputs["drive"]+params["b"])).astype(jnp.int32)
        return (out, None) if return_aux else out
```

Subclass `AbstractFactor` instead only if you implement `sample_with_references` yourself. Inherited: `sample_multiple(key, inputs, params, ..., n_samples)` which `vmap`s `sample` over split keys. Ports are declared with `jax.ShapeDtypeStruct` (shape and dtype, no data). Pbits are `int32`.

## DFG and Site

```python
Site(name, factor, parents, porting_fn, param_key, info_key, site_info)
sites = (Site("coin", CoinFactor(), (), (), "src", None, None),
         Site("out", ConditionalBit(), ("coin",), lambda outs: {"drive": outs[0].astype(jnp.float32)}, "cond", None, None))
graph = DFG(sites, {}, "out")                    # DFG(sites, input_ports, output_name)
params = graph.init_params(key)                  # dict keyed by DISTINCT param_key
x = graph.sample(key, {}, params)                # same call shape as any factor
```

- `porting_fn` is a tuple of port names (1:1 with `parents`) or a callable `parents -> input dict`.
- **Weight sharing = give sites the same `param_key`.** Verified: two `ConditionalBit` sites with `param_key="cond"` yield `params` keys `['src', 'cond']`.
- Check the graph against hand marginalisation: `P(out=1) = (1-p_c) sigmoid(b) + p_c sigmoid(w+b)`, sampled 0.546, exact 0.546 (40,000 draws).
- A `DFG` is itself a factor and nests as a `Site` factor. `DFGInfo(expose_site_outputs=..., entries=...)` passes runtime info; `gather_param_grads` exists on `AbstractDFG` for per-key gradient collection.
- Circuits are DFGs: `DiscretePCircuit.sample(key, {"in": bits}, params)` and `sample_multiple`; `HybridPCircuit.sample(key, {"discrete":..., "continuous":...}, params)`.

A `DFG` nests as the `factor` of a `Site` (run in `cookbook_12_study.py`): `Site("inner", inner_dfg, (), (), "inner_params", None, None)` makes the outer `params` tree `{'inner_params': {'src': ..., 'cond': ...}, 'cond2': ...}`; the nested sample matches the hand marginal (0.521).

## Composites

`ChainFactor` details (`composite_factors.py`): the base factor's input ports split into **feedback ports** (receive `feedback_porting_fn(previous step's main output)`; on the first step (index 0) they take the chain's initial state, i.e. the input you pass) and **broadcast ports** (the caller's value, unchanged at every step). `feedback_porting_fn` is a port-name string or a callable returning a dict; it is validated against `base.input_ports`. `sample` returns the final step's output; with `return_aux=True` the per-step aux is stacked on a leading axis of size `n_steps`. `slice_info=True` slices `info` per step.

`TiledFactor(base, n_tiles, weight_tied=True)`: `n_tiles` copies on a batched input; verified equal to `vmap` over split keys. `ChainFactor(base, n_steps, feedback_porting_fn="state", weight_tied=True)`: feeds the output back into the named port; verified equal to a manual loop over split keys.

## Hybrid regime switching (verified)

```python
gates = [psc.PditCycle(0, dims=3), psc.MixtureGaussianGate(sites=(0, 0), dims=(1,), num_components=3)]
circ = psc.HybridPCircuit(gates)
params = [jnp.array([2.0, -2.0]), {"means": means, "log_vars": log_vars}]   # PditCycle theta (2,), MoG dict
out = circ.sample(key, {"discrete": jnp.array([0], jnp.int32), "continuous": jnp.zeros(1)}, params)
```

Result (`cookbook_2_hybrid.py`): sampled `P(state=1) = 0.8653` (SE 0.0008, 200,000 samples) versus `softmax([0, 2, -2])[1] = 0.8668`. `HybridSampleSimulator` appears in notebook prose and the paper but **does not exist** in this source; use `circ.sample`.

## Exact Gaussian moments (verified against analytic)

```python
gs = psc.AffineGaussianSimulator()
comp = gs.build_circuit(hybrid_circ, thetas)
mom = gs.propagate(comp, jnp.zeros(2))                # .mean, .covariance, .site_moments
post = gs.condition(comp, {1: jnp.array([2.0])}, initial_continuous=jnp.zeros(2))
```

The input is a **point mass**; correlation comes from gates. Two-gate example: `x0 ~ N(0,1)`, then `x1 = 2 x0 + N(0, 0.25)`: covariance `[[1,2],[2,4.25]]`; conditioning on `x1=2` gives `E[x0|x1] = 0.941`, equal to the analytic `2*2/4.25`.

`JumpDiffusionGate(sites=(d, c), dims)` was run (`cookbook_7_api.py`): with `PNOT(theta=0)` as the firing pbit, `P(j=1) = 0.497`, mean `x | j=1` +3.002 for `jump_mean=3`, variance `x | j=0` 0.0099 for `diff_log_var = log 0.01`. `AffineGaussianSimulator.condition(..., jitter=...)` takes a regulariser; `jitter=1e-6` returned 0.99999976 on a deterministic observed site.

## Tractable factors: exact `log p` for score-function code (`torx/tractable_prob_factors.py`, run in `cookbook_7_api.py`)

- `AbstractHasLogProbability`: mixin with `log_probability(inputs, outputs, params, info=None, site_info=None, return_aux=False)`.
- `AbstractEnumerableOutputFactor` / `AbstractFiniteStateSpaceFactor`: finite output (and input) spaces with `get_nth_output_state`, `output_state_to_index`, `n_input_states`.
- `AbstractMatrixFactor`: declare the fields `input_states` and `output_states` (dicts of arrays enumerating the states) and implement `get_log_probability_matrix(params)` (rows = input states, columns = output states) and get `log_probability` for free; `TinyCategorical` in `cookbook_7` returns `[-1.3143, -0.5143, -2.0143]` equal to `log softmax(logits)`, and is both an `AbstractFiniteStateSpaceFactor` and an `AbstractHasLogProbability`.
- `DeterministicFactor(fn, input_ports, output_spec)`: output `fn(inputs, site_info)`, `log_probability` is 0 for the right output and `-inf` otherwise (xor example: 0.0 and -inf).

Score check (`cookbook_7_api.py`): `jax.grad(lambda pr: cat.log_probability({}, {"x": jnp.array(1)}, pr))(params)` gives `[-0.2687, 0.4021, -0.1334]`, equal to `onehot(1) - softmax(logits)`. Use these when you write a REINFORCE or Eq. 44 estimator over a DFG: the score of each stochastic site is `d log p(output | inputs) / d params`; deterministic sites contribute nothing.

## Writing a custom gate

Follow `AffineGaussianGate`/`MixtureGaussianGate` in `torx/psc/gates/_continuous.py` (nb12 `LangevinGate` in `examples/helpers/_langevin.py` is the nonlinear example): subclass `AbstractContinuousGate` (or `AbstractControlledContinuousGate`), define `sites`, `dims`, `init_params`, and `sample(key, inputs, params, ...)`; a nonlinear drift removes the closed-form Gaussian simulator, so reference values must come from quadrature (nb12 validates MALA on a 1-D target).
