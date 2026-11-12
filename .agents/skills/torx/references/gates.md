# Gate reference (checked against torx/psc/gates/ and by running)

Conventions
- Binary gates (K=2 branches): `G(theta) = (1 - sigmoid(theta)) I + sigmoid(theta) B`, `theta` shape `(1,)`. `theta=+20` is effectively deterministic, `-20` is the identity.
- Pdit gates with more than 2 branches use a **softmax over branches**: branch probabilities = `softmax([0, theta_1, ..., theta_{K-1}])`, `theta` shape `(K-1,)`. Verified: `PditCycle(0, dims=3)` with `theta=[1.0, -0.5]` gives `[0.2312, 0.6285, 0.1402]` (stay, forward, backward) equal to `softmax([0, 1, -0.5])`.
- State-vector order: row-major, **site 0 is the most significant bit**. Flipping site 0 of `|00)` lands on index 2 (`|10)`), verified in `cookbook_1_core.py`.
- Parameters: a list with **one entry per gate** (`len(thetas) == len(gates)`), not per distinct gate. `circuit.init_params(key)` returns this list.
- Input state of `StateVectorSimulator.density` is a probability vector of length `prod(dims)`; of `BranchingSimulator.sample` an integer array `(n_sites,)`.

## Binary gates (sites, branch B)

| Gate | Sites | Branch 1 (B), from `_binary.py` |
|---|---|---|
| `PNOT(i)` | 1 int | flip bit |
| `PReset(i)` | 1 int | set bit to 0 (`0->0, 1->0`) |
| `PSWAP([a,b])` | 2 | swap `01 <-> 10` |
| `PCNOT([c,t])` | 2 | flip t when c=1 (`10<->11`) |
| `PJUMP([a,b])` | 2 | `10 -> 01` one-way (not invertible), others fixed |
| `POR([a,b])` | 2 | bit a = OR(a,b), bit b = 0 (`01->10, 11->10`) |
| `PCopy([a,b])` | 2 | copy a into b (`01->00, 10->11`) |
| `PDEMUX([a,b])` | 2 | **branch 0 is not identity**: branch 0 resets bit b (`01->00, 11->10`); branch 1 copies a into b and resets a (`10->01, 11->01`) |
| `PMultiCNOT([c..., t])` | >=2 | flip last bit when all others are 1 |
| `PCSWAP([c,a,b])` | 3 | swap a,b when c=1 (Fredkin) |

Truth tables below were produced by a run (`cookbook_11_train.py`, theta=+20, site 0 = most significant bit): `PMultiCNOT([0,1,2])`: `110->111`, `111->110`, all others fixed; `PCSWAP([0,1,2])`: `101->110`, `110->101`, others fixed; `PDEMUX([0,1])`: `00->00 01->00 10->01 11->01`. `Mix(theta=pi/2)` rotates `(1,0)` to `(0,1)`; `GaussianNoiseGate(theta=log 0.25)` gives variance 0.248.

There is **no AND, XOR or multiplication gate**. `POR` clears its second bit, so it is not a plain OR on two wires.

## Pdit gates (`dims` = number of states)

Run tables (`cookbook_12_study.py`, theta=+20): `PditShift(dims=3)` maps state `i -> (i+1) mod 3` (`[1, 2, 0]`); `PditSWAP(dims=3)` maps `(i,j) -> (j,i)` for all 9 pairs.

| Gate | Constructor | theta | Meaning |
|---|---|---|---|
| `PditShift` | `PditShift(site, dims=d)` | `(1,)` | cyclic `i -> (i+1) mod d` with prob sigmoid(theta) |
| `PditCycle` | `PditCycle(site, dims=d)` | `(2,)` | stay / forward / backward, softmax([0, t1, t2]) |
| `PditSWAP` | `PditSWAP([a,b], dims=d)` | `(1,)` | swap two d-state pdits |

## Continuous (pmode) and hybrid gates

`sites` for continuous gates are continuous-site indices; `dims` is a tuple of per-site widths. Use inside `HybridPCircuit`.

| Gate | theta pytree (verified shapes for 2 scalar sites) | Action |
|---|---|---|
| `AffineGaussianGate(sites, dims)` | `{A:(2,2), b:(2,), log_var:(2,)}` | `x' = A x + b + N(0, exp(log_var))` |
| `Displace` | `(d,)` | deterministic `x' = x + theta` |
| `Scale` | `(d,)` | deterministic `x'_i = exp(theta_i) x_i` |
| `Mix` | scalar `()` | rotation by angle `theta` of **exactly two** scalar sites |
| `Diffuse` | `(d,)` | `x' = x + N(0, exp(theta))`; for diffusion coefficient `D` over time `t` set `theta = log(2 D t)` |
| `GaussianNoiseGate` | `(d,)` | `x' = x + N(0, exp(theta))` |
| `MixtureGaussianGate(sites=(d_site, c_site), dims=(w,), num_components=K)` | `{means:(K,w), log_vars:(K,w)}` | `x' = x + means[k] + N(0, exp(log_vars[k]))`, `k` = value of control pdit. It is **additive**, so p(x'|k) needs a known input (x=0 for a pure mixture) |
| `JumpDiffusionGate(sites=(d,c), dims)` | `{diff_log_var, jump_log_var, jump_mean}` | binary-firing jump; different model from regime switching |

Caveats from the notebooks and source:
- Deterministic channels use `log_var = -inf` (nb11 `_det_log_var`); the notebook advises clamping before training. Verified (`cookbook_7_api.py`): through `AffineGaussianSimulator` the gradient is finite (`d/dA = 4.0`, `d/dlog_var = 0.0`); through `HybridPCircuit.sample` `d/dlog_var` is **NaN** (A and b finite). Clamp to about -30 when differentiating `log_var` by sampling. `condition` on a deterministic observed site may raise an `EquinoxRuntimeError` asking for `jitter`; `condition(..., jitter=1e-6)` returned 0.99999976 in our example.
- Shared `theta` across time steps in nb11 is inference-time reuse (same pytree entry repeated), not trainable tying; `reps` or `Site.param_key` are the tying mechanisms (see factors-dfg-hybrid.md).
- `PISING(sites, ...)`: `theta=[J,h1,h2,beta,dt]`, Glauber `exp(dt*Q)`, **state-vector only**; `BranchingSimulator` raises. Equilibrium reached by `dt ~ 20` in our test (50 used). `J>0` ferromagnetic (aligns), `J<0` antiferromagnetic.

## Hybrid register

`HybridPCircuit.sample(key, {"discrete": int array, "continuous": float array}, params)` returns the same dict. `sample_multiple(..., n_samples=N)` stacks. Discrete and continuous site indices are separate namespaces.
