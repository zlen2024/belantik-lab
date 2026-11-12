# How to explain a Torx program (protocol and two worked examples)

## Protocol: eight questions, in order

1. **Sites.** List every wire, its type (pbit/pdit/pmode) and its index namespace (discrete and continuous are separate in a hybrid circuit). Say how many states the register has (`2^n` for n pbits) and whether the exact simulator is feasible.
2. **Gates in order.** For each gate: name, sites, the branch operation `B` (gates.md), and the parameter shape. Say what happens at `theta=+20` and `theta=-20`.
3. **Kernel.** Convert `theta` to a probability: `p = sigmoid(theta)`; for K>2 branches `softmax([0, theta])`. Write the column-stochastic matrix or the update rule in one line.
4. **Equation to code.** Name the paper or notebook equation and show the line of code that implements it (for example Eq. 17 = `G(theta) = (1-sigma)I + sigma B`).
5. **Simulator and why.** Exact `StateVectorSimulator` (small, differentiable) vs sampled `BranchingSimulator` (large, gradient by parameter shift) vs `AffineGaussianSimulator` (exact Gaussian moments). State the input format (probability vector vs integer bits vs point mass).
6. **Readout and estimator.** Which `f` is averaged; exact expectation or Monte Carlo mean; the standard error (`1/sqrt(N)`) and, for the p-bit dot product, the closed form in estimator-math.md. Report a finite-sample floor when comparing distributions.
7. **Gradient path.** Which `diff_method`, why it is unbiased or not, and whether the loss uses only `expval_all`. If the loss needs a joint statistic, say a score-function term is required.
8. **Where the code departs from the ideal equation.** Finite reset strength, Trotter bias, clipped means, shared-vs-trained parameters, exact density used inside a "sampled" recipe. Mark each as source, own run, or assumption.

Keep numbers traceable: quote the script that produced them.

## Worked example A: three gates, by hand versus code (`cookbook_1_core.py`)

```python
circ = psc.DiscretePCircuit([psc.PNOT(0), psc.PCNOT([0, 1]), psc.PNOT(1)])
thetas = [0.3, -0.4, 0.8]       # sigmoid -> p = 0.5744, 0.4013, 0.6900
```

1. Sites: two pbits, state space 4, start `|00)`; exact simulation is trivial.
2. `PNOT(0)` flips site 0 with p1 = 0.5744; `PCNOT([0,1])` flips site 1 only when site 0 is 1, with p2 = 0.4013; `PNOT(1)` flips site 1 with p3 = 0.6900.
3. Propagate the marginal of site 1 by hand: after the CNOT `P(b1=1) = p1 p2 = 0.2305`; after the final NOT `P(b1=1) = 0.2305 (1-p3) + 0.7695 p3 = 0.6024`.
4. Equation: each gate is Eq. 17 with `B` = NOT / CNOT; the circuit is the product of three column-stochastic matrices (Eq. 16).
5. Readout `f(Y) = Y_1`. Gradient by the chain rule, using `sigma'(theta) = sigma(1-sigma)`:
   - `dE/dtheta3 = sigma'(0.8) (1 - 2*0.2305) = 0.2139 * 0.5389 = 0.1153`
   - `dE/dtheta2 = p1 sigma'(-0.4) (1 - 2 p3) = 0.5744 * 0.2403 * (-0.38) = -0.0524`
   - `dE/dtheta1 = sigma'(0.3) p2 (1 - 2 p3) = 0.2445 * 0.4013 * (-0.38) = -0.0373`
6. These equal the `jax.grad` of the exact state-vector loss (`-0.0373, -0.0524, 0.1153`), and the three sampled methods match them in the mean (gradients.md table). Explaining the code means showing the hand number and the printed number side by side.
7. Departures from ideal: none for the exact path; the sampled path adds Monte Carlo noise, with `param_shift_single` about 3x noisier than `param_shift_inf` here.

## Worked example B: one chromatic-Gibbs site update (`cookbook_5_gibbs.py`, paper Eq. 58-60)

1. Sites: one Ising spin as a pbit; the ring's other spins are host-side numbers that enter only through the local field.
2. Gates: `PReset` then `PNOT`. `PReset(theta=+20)` sets the bit to 0 with probability near 1; `PNOT(theta)` then flips a 0 to 1 with `sigmoid(theta)`.
3. Kernel: `P(sigma=1) = sigmoid(theta)`. Choose `theta = 2 beta l_i`, `l_i = h_i + sum_j J_ij (2 sigma_j - 1)`, to realise the Gibbs conditional `1/(1+e^{-2 beta l_i})`.
4. Check on the exact simulator: start 0 and start 1 both give `P(1) = 0.8022 = sigmoid(2*1.0*0.7)`. This also shows the reset removes dependence on the previous value.
5. Full sampler: two colour layers on a two-colourable ring, 6000 chains, 240 sweeps. Sites in one colour are conditionally independent given the other colour, so they update in parallel.
6. Estimator: empirical histogram over the 256 states against the exact Boltzmann law; TV 0.0457. A histogram of 6000 draws from an **exact** sampler has TV 0.0468 on average, so the sampler is consistent with zero bias. Quote both numbers.
7. Why the per-bond alternative fails (notebook 06 and paper p. 13): a `PISING` bond kernel uses its own coupling and half of each endpoint's field and omits the energy shared with the neighbour outside the pair, so the composed sweep is not a Glauber kernel conditioned on the rest of the ring and need not satisfy detailed balance (TV 0.402 in nb06). The paper adds that the composed flip rate is a sum of per-neighbour sigmoids where the Glauber rate has the full local field inside one sigmoid. Any explicit per-site formula beyond that is untested here.
8. Departures: `PReset(+20)` is not infinite reset (error about `1 - sigmoid(20) = 2e-9`); the local field is computed on the host; notebook 06 uses finite reset 12 in its one-site probe.

## Worked example C: a DFG with a custom factor (`cookbook_3_factors.py`)

1. Sites: `coin` (no parents, params under `"src"`) and `out` (parent `coin`, params under `"cond"`). `DFG(sites, {}, "out")`. The DAG is evaluated parent before child, one call `graph.sample(key, {}, params)`.
2. Factors: `CoinFactor` draws `Bernoulli(sigmoid(bias))`; `ConditionalBit` draws `Bernoulli(sigmoid(w*drive + b))`. `porting_fn` casts the parent bit to the float `drive` input.
3. Hand computation with `bias=0.4, w=2, b=-1`: `P(coin=1) = sigmoid(0.4) = 0.5987`; `P(out=1) = 0.4013*sigmoid(-1) + 0.5987*sigmoid(1) = 0.4013*0.2689 + 0.5987*0.7311 = 0.5456`.
4. Printed: sampled 0.546, exact 0.546 (40,000 draws).
5. Simulator: none. A DFG samples by walking the DAG, so there is no state vector; exactness comes from hand marginalisation.
6. Parameter sharing: give two sites the same `param_key`; `init_params` then returns one slice per distinct key (`['src', 'cond']`).
7. Departures: factors carry parameters outside the object; `info` is optional runtime input (`DFGInfo` + `Site.info_key`; `cookbook_7_api.py` shows scale 0 -> P=0.503, scale 3 -> 0.953 against 0.500 and 0.953).

## Worked example D: hybrid regime switching and a Gaussian posterior (`cookbook_2_hybrid.py`)

1. Sites: discrete site 0 (3-state pdit) and continuous site 0 (1-d pmode); the two index namespaces are separate.
2. Gates: `PditCycle(0, dims=3)` with `theta=[2, -2]` gives `softmax([0, 2, -2]) = [0.117, 0.867, 0.016]` for stay, forward, backward; from state 0 forward lands in state 1. `MixtureGaussianGate(sites=(0,0), dims=(1,), num_components=3)` then adds `means[k]` and noise `exp(log_vars[k])` where `k` is the new discrete state.
3. Hand number: `P(state=1) = 0.8668`; printed 0.8653 with SE 0.0008 (200,000 draws), 1.9 SE away.
4. Gaussian part: gate 1 draws `x0 ~ N(0,1)`, gate 2 sets `x1 = 2 x0 + N(0, 0.25)` and keeps `x0`. Covariance by hand: `Var x0 = 1`, `Cov = 2`, `Var x1 = 4.25`; printed `[[1,2],[2,4.25]]`.
5. Posterior: `E[x0 | x1=2] = Cov/Var(x1) * 2 = 2/4.25 * 2 = 0.941`; `condition` printed posterior mean 0.941 (`cookbook_2_hybrid.py`). Autodiff (`cookbook_7_api.py`) of this posterior mean w.r.t. `A21` and `log_var` matches finite differences (-0.4152 vs -0.4153, -0.0554 vs -0.0554).
6. Simulator and why: the Gaussian simulator is exact and differentiable because every gate is affine-Gaussian; the mixture gate and any custom nonlinear gate (e.g. `OUStep` in `cookbook_7_api.py`, checked against the analytic AR(1) mean 0.878 and variance 0.4667 after 20 steps) need sampling instead.
7. Departures (`cookbook_7_api.py`): the simulator input is a point mass, so correlation must come from gates; `log_var = -inf` is finite through the exact simulator but gives NaN `d/dlog_var` through `HybridPCircuit.sample`, so clamp to about -30 there; the discrete `PditCycle` choice has zero gradient (pathwise gradients cover only the continuous part).

## Worked example E: a training loop, traced (`cookbook_11_train.py` part 1)

```python
def step(th, st, key):
    g = -grad_fn(th, key)                      # minimise -E[bit 1]
    upd, st = opt.update(g, st); return optax.apply_updates(th, upd), st
```

1. Inputs: `th` (3 logits, the only trainable pytree), the optimiser state, and a fresh PRNG key every step (`key, k = jax.random.split(key)`). Reusing one key would replay the same samples every step and lock in one noise realisation.
2. Forward: `build_circuit(circ, [th[0:1], th[1:2], th[2:3]])` rebuilds the compiled circuit from the current logits, then `expval_all(comp, x0, key)[1]` returns the sampled mean of bit 1 (`S` samples).
3. Backward: `jax.grad` runs the simulator's custom rule: parameter shift (`diff_method`) re-runs shifted circuits, REINFORCE instead weights `d log p` by `f - baseline`. The gradient has the same structure as the input list/array.
4. Hand number before training for the 3-gate chain: `E = 0.6024` for `theta = [0.3, -0.4, 0.8]` (example A). After 300 Adam steps the **exact** objective is 0.9986 with the exact, `_inf`, `_filter` and REINFORCE gradients; theta about `[-4.1, -4.4, 6.7]`, so every gate is nearly deterministic (the first two near the identity, the last a NOT).
5. Why the sampled loss is noisy: its sd is `1/sqrt(S)`; never judge progress by one sampled value. Print the exact objective (state vector) on a tiny circuit, the gradient against the exact gradient, and the sampled/exact gradient ratio.
6. Where it would go wrong: a loss nonlinear in `E_hat` (squared error) can have a biased gradient (`param_shift_single` bias `2 Var(f)/S` is at most 0.5/S; at S=20 it is larger than two of the three gradient components, gradients.md), a discrete gate in a hybrid circuit gets zero gradient, `reps != 1` with `_inf`/`_single` raises.

## Worked example F: a custom continuous gate, line by line (`OUStep` in `cookbook_7_api.py`)

```python
class OUStep(AbstractContinuousGate[dict, tuple[int, ...]]):
    _draw_label = "OU"
    sites: HybridSites = eqx.field(converter=_sites)
    dims: tuple[int, ...]
    def init_params(self, key):
        return {"a": jnp.array(0.1), "mu": jnp.array(1.0), "sigma": jnp.array(0.3)}
    def sample(self, key, inputs: Mapping[str, Any], params, info=None, site_info=None, return_aux=False):
        x = inputs["continuous"]
        out = x + params["a"] * (params["mu"] - x) + params["sigma"] * jax.random.normal(key, x.shape)
        return (out, None) if return_aux else out
```

1. Sites: `sites` is converted to the `{"discrete": [], "continuous": [...]}` form; `dims` gives each continuous site's width; these two fields are what a `HybridPCircuit` reads.
2. Kernel: `x' = x + a (mu - x) + sigma * N(0,1)`, an AR(1) step with fixed point `mu`; the whole gate is the sampler (there is no transition matrix).
3. Interface: `inputs` is a dict keyed by `"continuous"` (and `"discrete"` for controlled gates); `return_aux` toggles a `(out, aux)` return. The key is split by the caller; use it exactly once per random draw.
4. Equation to code: after `n` steps from `x = 0`, `E[x] = mu (1 - (1 - a)^n)` and `Var[x] = sigma^2 (1 - (1 - a)^{2n}) / (1 - (1 - a)^2)`.
5. Run: `HybridPCircuit([OUStep(...)], reps=20)`, 40,000 samples: mean 0.875 (exact 0.878), variance 0.4623 (exact 0.4667; within sampling error for 40,000 samples).
6. Simulator: sampling only. Without an `affine_parameters` method the gate has no `AffineGaussianSimulator` channel; a nonlinear drift (nb12 `LangevinGate`) never has one, so validate against quadrature or an analytic limit as above.
7. Differentiability: gradients w.r.t. `a`, `mu`, `sigma` flow pathwise through `HybridPCircuit.sample` (reparameterised normal); there is no discrete branch to miss.

## Worked example G: reviewing an unfamiliar (buggy) program with the checklist (`cookbook_12_study.py` part 2)

```python
circ = psc.DiscretePCircuit([psc.PNOT(0), psc.PNOT(1)])
th = [jnp.array([0.0])]                                   # (1)
comp = sv.build_circuit(circ, th)
dens = jax.jit(sv.density)(comp, x)                       # (2)
loss = bsim.sample(bsim.build_circuit(circ, th2), x0, key).mean()   # (3)
K[3] += 0.8 * q;  w = softmax(K @ q / np.sqrt(d))         # (4) attention test
```

1. Sites and parameters: two gates need two thetas. Real message: `ValueError: expected 2 thetas aligned with circuit.gates, got 1.` Fix: one `theta` per gate, shape `(1,)` each.
2. Simulator and compilation: `jax.jit` over a compiled circuit raises `TypeError: Shapes must be 1D sequences of concrete values of integer type, got (JitTracer(~int32[]), ...)`. Fix: `equinox.filter_jit(sv.density)`, or rebuild the circuit inside the jitted function.
3. Gradient path: a loss built from `sample()` has gradient `[0.0, 0.0, 0.0]` with **no error**. Fix: use `expval_all` (or the exact density) and check against the state-vector gradient.
4. Estimator and test design (`cookbook_12_study.py`): with `K[3] += 0.8 q` the true softmax has max weight 0.999914 (0.999995 in the original `z1_attention.py` demo; the value depends on the random keys), so any TV test reads 0.0000 at every N. Fix: spread the logits (max weight about 0.36 in `cookbook_4`) and assert the max weight is below 0.99, as `z1_attention.py` now does (`assert true_w.max() < 0.99`).
5. A fifth cue seen elsewhere: `PISING` with `BranchingSimulator` raises `ValueError: BranchingSimulator only supports AbstractKBranchGate lookup-table gates; unsupported gates: PISING`; use the state-vector simulator.

The order matters when explaining: first the cheap structural errors (1, 2, 5), then the silent ones (3, 4), which are the ones that produce plausible wrong numbers.

## Worked example H: conditioning a Gaussian circuit with hand arithmetic (`cookbook_12_study.py` part 3)

1. Program: `x0 ~ N(0,1)` (gate 1), then `x1 = 2 x0 + N(0, 0.25)` with `x0` kept (gate 2); observe `x1 = 2`.
2. Call: `gs.condition(comp, {1: jnp.array([2.0])}, initial_continuous=jnp.zeros(2))`; the dict keys are **site indices** of the observed continuous sites; `initial_continuous` is the point-mass input; `jitter` regularises a singular observed covariance.
3. By hand: `Var x1 = 4 + 0.25 = 4.25`, `Cov(x0, x1) = 2`. Posterior `mean = Cov/Var(x1) * 2 = 4/4.25 = 0.9412`; posterior `variance = 1 - Cov^2/Var(x1) = 1 - 4/4.25 = 0.0588`.
4. Printed: posterior mean `[0.9412]`, covariance `[[0.0588]]`, observed sites `[1]`. The returned object holds only the unobserved (queried) sites; this is why a one-element array comes back.
5. Departures: exact because every gate is affine-Gaussian; a deterministic observed site may need `jitter` (`cookbook_7_api.py`: `jitter=1e-6` gave 0.99999976).

## Worked example I: interpreting a result (`cookbook_13_bias_theory.py`, `cookbook_12_study.py`)

Question: "I minimise `(E_hat - 0.8)^2` with `param_shift_single` at `S=20` and the exact objective converges to 0.75, not 0.80. Is training broken?"

1. Identify the loss: it is nonlinear in the sampled expectation `E_hat`, so a gradient estimator that reuses the primal samples is suspect (Jensen trap).
2. Identify the estimator: `param_shift_single` multiplies `grad_out = 2 (E_hat - t)` by a difference built from the same samples, so its expected gradient has an extra term `2 Var(f)/S` (gradients.md law).
3. Compute the predicted fixed point. For a single gate `E* = t - 1/S = 0.8 - 1/20 = 0.7500`; the run gives 0.7509 (S=20; 0.7794 against 0.7800 at S=50). For several gates there is no closed form; combine the exact gradient with the law (`2 Var(f)/S` on every gate for a quadratic loss) and simulate the deterministic optimiser: on the 3-gate chain this ends at 0.7500 (`cookbook_13_bias_theory.py`), and the sampled single-batch runs gave 0.7367 with SE 0.0128 over 6 seeds (`cookbook_12_study.py`), about one SE away.
4. Rule out the alternatives: `_inf` converges to 0.8002 and `_filter` to 0.8434 (predicted 0.8421) at S=20 on the single-gate problem (opposite-sign shift for this single gate; on other circuits the `_filter` sign can match `_single`'s, gradients.md), so the objective, learning rate and circuit are not the cause.
5. Fix and verify: for this quadratic loss use `_inf` (unbiased) or the two-batch product (on the 3-gate chain 0.8035 against 0.7367 for the single batch), or raise `S` until `1/S` is below the tolerance. For a different nonlinear loss the advice changes: all methods are biased at O(1/S) (per-method table in gradients.md; cubic and exp checked), a polynomial of degree k needs k independent batches. State what was verified (closed forms, the cubic and exp losses, K=2 gates) and what was not (K>2 gates for `_single`, deeper circuits beyond the 6-gate example).

## Failure checklist when explaining or reviewing someone else's Torx code

- Working directory is `master/` (the `torx/` folder shadows the package: `ImportError: cannot import name 'psc'`).
- A compiled circuit passed into `jax.jit`: `StateVectorSimulator.density` raises `TypeError ... JitTracer` and `BranchingSimulator.expval_all` raises `TracerBoolConversionError`; use `equinox.filter_jit` (Example G) or build the circuit inside the jitted function.
- `len(thetas) != len(gates)`: `ValueError: expected N thetas aligned with circuit.gates` from both simulators. A scalar theta where `(1,)` is expected builds without error; check shapes yourself.
- `PISING` with `BranchingSimulator`: `ValueError: BranchingSimulator only supports AbstractKBranchGate lookup-table gates; unsupported gates: PISING`.
- `jax.grad` through `sample()` returns zeros silently; the loss must use `expval_all`.
- `reps != 1` with `param_shift_inf` or `param_shift_single` raises `NotImplementedError`; K>2 gates with `param_shift_single` also raise.
- Site order: site 0 is the most significant bit of a state-vector index; the discrete and continuous namespaces of a hybrid circuit are separate.
- A gate list reused with different dims on one site raises a dimension-mismatch `ValueError`.
- Windows console: `draw_circuit` prints box-drawing characters and fails on a cp1252 console; set `PYTHONIOENCODING=utf-8` or `sys.stdout.reconfigure(encoding="utf-8")`.
- Saturated softmax or a TV below the finite-sample floor makes a test meaningless.
- Same PRNG key reused across steps or across what should be independent batches (identical samples; the two-batch product trick needs two keys).
- A discrete gate inside a hybrid circuit being trained by `jax.grad(sample)` alone: its parameters silently get zero gradient.
- A squared loss on one sampled expectation: check bias against the two-batch product (gradients.md).

## Style rules for explanations

- Say what is verified: "ran", "from the whitepaper p. N", "from nb06 prose, not re-run".
- Prefer a small number the reader can recompute to a big table.
- Never call a sampled result "unbiased" without naming the estimator and its conditions (positivity, `reps`, branch counts).
