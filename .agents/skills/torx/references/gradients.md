# Gradients and training in Torx

## The one rule

Differentiable readouts are **per-site expectations**: `StateVectorSimulator.expval/expval_all` (exact, autodiff) and `BranchingSimulator.expval/expval_all` (sampled, parameter-shift). `BranchingSimulator.sample()` returns bits and is **not differentiable**: `jax.grad` through a loss built from `sample()` returns exact `[0., 0., 0.]` **with no error** (verified), so an agent can train nothing silently. Always check a new gradient against the exact state-vector gradient on a tiny circuit. Any loss that needs a joint statistic (`E[s_i s_j]`, a cut value, an attention score) cannot be written as a function of `expval_all` and needs either (a) the exact density from `StateVectorSimulator` (only for small `n`), or (b) a score-function estimator you write yourself (below).

## Idiom (verified, `examples/cookbook_1_core.py`)

**`jit` rules.** `jax.jit(loss)` over `build_circuit` + `expval_all` works (the circuit is rebuilt inside the traced function). Passing an already compiled circuit into `jax.jit` fails: `jax.jit(sv.density)(comp, x)` raises `TypeError: Shapes must be 1D sequences of concrete values of integer type, got (JitTracer(~int32[]), ...)` because the compiled circuit's dims are not static. Use `equinox.filter_jit(sv.density)` (treats non-array leaves as static) or rebuild the circuit inside the jitted function.

```python
bsim = psc.BranchingSimulator(num_samples=4000, diff_method="param_shift_inf")
def loss(thetas, key):
    comp = bsim.build_circuit(circ, thetas)        # building inside jit is fine
    return bsim.expval_all(comp, x0, key)[1]
g = jax.jit(jax.grad(loss))(thetas, key)           # list of (1,) arrays, same structure as thetas
```

This is exactly how `torx/tests/test_gradients.py` checks every binary gate against the state-vector gradient. Compiling outside the loop is only a speed choice when `thetas` do not change.

## The three `diff_method` options

Notes: the `reps != 1` errors are raised only when differentiating (the forward `expval_all` works); the paper (p. 8) notes only gates downstream of `j` need resampling, but Torx re-runs the full circuit for each shifted or deterministic branch. (source `psc/simulation/sampled.py`, measured)

Measured on a 3-gate circuit, `num_samples=4000`, 40 repeats, gradient of `E[bit 1]` w.r.t. each theta (exact `[-0.0373, -0.0524, 0.1153]`):

| method | mean (3 thetas) | sd over repeats | cost | restrictions |
|---|---|---|---|---|
| `param_shift_inf` | -0.037, -0.053, 0.115 | 0.003, 0.003, 0.002 | `(1 + G*K_max)` x S samples (every gate is padded to the largest branch count) | unbiased; **raises for `reps != 1`** |
| `param_shift_single` | -0.037, -0.051, 0.114 | 0.009, 0.009, 0.011 | `(1 + G)` x S samples | K=2 gates only; **raises for `reps != 1`**; uses `theta* = -ln((1+e^{-theta})^2 - 1)` (paper Eq. 39) |
| `param_shift_filter` | -0.037, -0.053, 0.114 | 0.004, 0.003, 0.005 | S samples (one forward pass) | supports `reps > 1`; **ratio estimator** (conditional means from the same samples, empty branches set to 0): finite-N bias with rare branches and a nonlinear-loss bias (formulas below); higher variance than `_inf` at equal S |

Rule of thumb (one 3-gate circuit): at equal `S` `_inf` has the lowest variance and `_single` is noisier at `p=1/2` (sd 0.0215 against 0.0054 at S=1000, `cookbook_6_gradients.py`); at equal total circuit runs the ranking depends on `theta` (table below: `_inf` wins at -4 and +2, `_filter` at -2 and 0). `_filter` is also the only choice for repeated circuits.

## Systematic evidence (`examples/cookbook_6_gradients.py`, all numbers from that run)

**Unbiasedness.** K=2 chain with `theta=[t,t,t]` for `t in {-5,-2,0,2,5}`, N=4000, 30 repeats, three methods: the largest `|mean - exact|` was 0.0010 (`_inf`), 0.0052 (`_single`, at `t=0`, 2.5 SE) and 0.0010 (`_filter`). 45 comparisons were made, so one entry near 2.5 SE is expected by chance; no systematic bias is visible. At `t=+/-5` the gates are nearly deterministic and all sds collapse to about 0. K=3 (`PditCycle`, `theta=[t,-t]`, `t in {-3,0,3}`, loss = mean index): `_inf` exact to 4 decimals; `_filter` within 0.0013 (about 1.4 SE); `_single` raises `NotImplementedError` (K>2).

**Variance: derived and checked (`cookbook_9_variance.py`).** With `S = num_samples`, `p = sigmoid(theta_j)` and readout `f`:

- `param_shift_single` is exactly paper Eq. 40 (checked in `cookbook_9_variance.py`): `sd = sqrt((Var f at theta + Var f at theta^-)/S)` with `sigma(theta^-) = p^2`. Single `PNOT`, `theta=0`, `S=2000`, 300 repeats: predicted 0.0148, measured 0.0150.
- `param_shift_inf` is `g_j = p(1-p)(E1 - E0)` where `E_b` is the mean of `f` with gate `j` forced to branch `b` (deterministic branch evaluations): `sd = p(1-p) sqrt((Var_0 f + Var_1 f)/S)`. For a single `PNOT` both branch variances are 0, so the sd is 0 (measured 0.0000). The factor `p^2 (1-p)^2` is 1/16 at `p = 1/2`, which explains why `_single` is about 4x noisier at `theta=0`. 3-gate chain, `S=1000`, 300 repeats, per-gate sd predicted vs measured: `_inf` 0.0056/0.0056, 0.0056/0.0060, 0.0048/0.0047; `_single` 0.0224/0.0231, 0.0224/0.0224, 0.0220/0.0231.
- `param_shift_filter` is a ratio estimator; single-gate closed form `sd = p sqrt(p(1-p)/S)` (part 1b of the same script; details in the `_filter` paragraph below).
- All three scale as `1/sqrt(S)` (sd ratio S=1000 vs 4000, `cookbook_6_gradients.py`: 1.92 for `_inf`, 1.88 for `_single` and 2.05 for `_filter`).

**Cost-normalised comparison (`cookbook_9_variance.py` part 3; equal budget T = 28,000 circuit runs per gradient, 3 gates, K=2, `theta=0`, mean sd over the 3 gates, 100 repeats).** Runs per gradient from `sampled.py`: `_inf` `(1 + G*K) S`, `_single` `(1 + G) S`, `_filter` `S`.

| method | S used | sd | max abs bias |
|---|---|---|---|
| `param_shift_inf` | 4,000 | 0.0027 | 0.0006 |
| `param_shift_single` | 7,000 | 0.0082 | 0.0006 |
| `param_shift_filter` | 28,000 | 0.0013 | 0.0001 |
| REINFORCE (plain JAX, batch-mean baseline) | M=28,000 | 0.0014 | 0.0004 |

At equal cost `_filter` and REINFORCE are best on this circuit and `_single` is worst; the earlier "`_inf` has the lowest variance" holds only at equal `S`, not at equal cost. `_filter` and REINFORCE results are for a 3-gate circuit with `p = 1/2`; do not extrapolate to deep circuits or extreme `p` without rerunning the script.

**Equal budget depends on `theta` (T = 28,000 runs, 3-gate chain, 60 repeats, RMSE = sqrt(bias^2 + sd^2) averaged over the gates; `cookbook_9_variance.py` section 5):**

| theta | `_inf` | `_single` | `_filter` | best |
|---|---|---|---|---|
| -4 | 0.00005 | 0.00222 | 0.00012 | `_inf` |
| -2 | 0.00074 | 0.00506 | 0.00072 | `_filter` (tie with `_inf`) |
| 0 | 0.00272 | 0.00890 | 0.00152 | `_filter` |
| +2 | 0.00089 | 0.00711 | 0.00120 | `_inf` |

So **the ranking is not universal**: `_inf` wins when `p` is far from 1/2, `_filter` only near `p = 1/2` (a tie at -2), and `_single` never wins. **Choosing a method:** use `_inf` for a circuit with few gates when `reps == 1` (for a nonlinear loss it is the least biased of the three but not unbiased: see the per-method table); use `_filter` for repeated circuits (`reps > 1`) or many gates (its cost does not grow with the number of gates), making sure every branch is sampled many times; `_single` is unbiased for a readout linear in the sampled expectation but biased for nonlinear losses (law below) and never won the equal-cost comparison, so prefer the others; for joint statistics use REINFORCE.

**Depth.** The `_inf` and `_single` formulas also hold on longer chains (mixed `PNOT`/`PCNOT`, `theta=0`, `S=1000`, 200 repeats): depth 6 `_inf` predicted 0.0056 measured 0.0054, `_single` 0.0224 / 0.0226; depth 10 `_inf` 0.0056 / 0.0056, `_single` 0.0224 / 0.0223.

**Jensen trap (loss nonlinear in the sampled expectation; `cookbook_11_train.py` part 2).** `E[(E_hat - t)^2] = (E - t)^2 + Var(E_hat)`, and when the gradient estimator is built from the **same samples** as `E_hat` the product `grad_out * dE_hat` is correlated and biased. Measured on `(E_hat - 0.8)^2`, 3000 repeats per cell, true gradient `[0.0147, 0.0207, -0.0456]`, bias = mean - true:

| S | `_inf` | `_single` | `_filter` |
|---|---|---|---|
| 20 | -0.0000, -0.0006, +0.0006 | +0.0226, +0.0230, +0.0232 (SE 0.0011-0.0015) | +0.0001, +0.0001, -0.0040 (SE 0.0008 on the last) |
| 100 | +0.0002, +0.0001, -0.0000 | +0.0052, +0.0050, +0.0051 (SE 0.0004-0.0006) | +0.0002, +0.0001, -0.0011 (SE 0.0003) |
| 1000 | 0.0000 (all three, SE <= 0.0001) | +0.0004, +0.0005, +0.0005 (SE 0.0001-0.0002) | -0.0000, +0.0000, -0.0001 |

**Law of the bias and its scope (`cookbook_11_train.py` part 2b, `cookbook_13_bias_theory.py` part 1).** For `param_shift_single` the gradient estimator is `L'(E_hat) * (E_hat - E_tilde'_j)`, where `L'(E_hat)` is built from the **same primal samples** as `E_hat` and `E_tilde'_j` comes from the circuit with gate `j` moved to `theta_j^-` (`sigma(theta^-) = sigma(theta)^2`). Expanding `L'(E_hat) = L'(E) + L'' d + L''' d^2 / 2` in the fluctuation `d = E_hat - E` (variance `Var(f)/S`) gives, to second order in `d`,

`bias_j = [ L''(E) + (1/2) L'''(E) (E - E'_j) ] Var(f) / S`,

where `E'_j` is the exact expectation of the readout with gate `j` shifted. For the quadratic loss `(E_hat - t)^2` we have `L'' = 2`, `L''' = 0`, so the bias is `2 Var(f)/S` and **identical on every gate**; for a bit readout `Var(f) = E(1-E) <= 1/4`, so it is at most `0.5/S`. For a loss with `L''' != 0` the bias is **gate dependent** through `E'_j`. Printed hand check for the 3-gate chain (quadratic): `E = 0.6024`, `2 E(1-E) = 0.479`; predicted `S=20` 0.02395 against measured 0.0226-0.0232; `S=100` 0.00479 against 0.0050-0.0052; `S=1000` 0.00048 against 0.0004-0.0005. At `S=20` the biased estimator overestimates the first two components 2.7x and 2.1x (printed means 0.040 and 0.044 against true 0.0147 and 0.0207) and halves the third (-0.022 against -0.0456, ratio 0.5); no component changes sign; at `S=1000` the bias shrinks like `1/S` but is not gone.

Per-gate check with independent PRNG keys, 20,000 repeats, bias = mean - true; `predicted` uses the formula above with printed `E'_j`; SE is the maximum over gates:

| circuit, readout, loss | S | predicted per gate | measured per gate | SE |
|---|---|---|---|---|
| 4-gate, bit 1, quad | 40 | 0.0120 0.0120 0.0120 0.0120 | 0.0121 0.0123 0.0123 0.0116 | 0.0004 |
| 4-gate, bit 1, quad | 160 | 0.0030 0.0030 0.0030 0.0030 | 0.0030 0.0028 0.0030 0.0030 | 0.0002 |
| 6-gate mixed, bit 1, quad | 40 | 0.0062 (all six) | 0.0061 0.0051 0.0052 0.0067 0.0054 0.0056 | 0.0009 |
| 6-gate mixed, sum of bits, quad | 40 | 0.0154 (all six) | 0.0155 0.0153 0.0155 0.0157 0.0157 0.0155 | 0.0005 |
| 6-gate mixed, sum of bits, quad | 160 | 0.0039 (all six) | 0.0037 0.0039 0.0041 0.0040 0.0043 0.0042 | 0.0002 |
| 6-gate mixed, bit0 - bit1, quad | 40 | 0.0197 (all six) | 0.0200 0.0200 0.0196 0.0195 0.0208 0.0200 | 0.0008 |
| 6-gate mixed, bit0 - bit1, quad | 160 | 0.0049 (all six) | 0.0047 0.0047 0.0048 0.0049 0.0049 0.0050 | 0.0004 |
| 6-gate mixed, bit 1, cubic `e^3` | 40 | 0.0028 0.0027 0.0027 0.0027 0.0027 0.0017 | 0.0029 0.0029 0.0029 0.0029 0.0029 0.0019 | 0.0001 |
| 6-gate mixed, bit 1, cubic `e^3` | 160 | 0.0007 0.0007 0.0007 0.0007 0.0007 0.0004 | 0.0007 0.0007 0.0007 0.0007 0.0007 0.0004 | 0.0000 |
| 6-gate mixed, bit 1, `exp(-3e)` | 40 | 0.0179 0.0181 0.0180 0.0180 0.0182 0.0209 | 0.0176 0.0187 0.0170 0.0185 0.0196 0.0192 | 0.0014 |

So the law holds for non-bit readouts (with the variance of the readout) and the corrected formula reproduces the **gate-dependent** pattern of the cubic loss, including the last gate (`POR`, whose shifted readout moves most: `E'_6` far from `E`). The earlier "first-order statement" explanation for the last gate was wrong: the deviation is the same order in `1/S` and comes from the `L'''` term. Earlier single-seed rows with 4000 repeats showed all gates low together because they shared PRNG keys (one correlated noise realisation).

**Bias of each method for a nonlinear loss, K=2 gates (`cookbook_13_bias_theory.py` parts 1, 1b, 1c).** Let `g_j = dE/dtheta_j = E - E'_j`.

| method | bias on gate `j` | source of each term |
|---|---|---|
| `param_shift_inf` | `(1/2) L'''(E) g_j Var(f)/S` | no covariance term (branch evaluations use independent keys), but `E[L'(E_hat)] = L'(E) + L''' Var/(2S)` still differs from `L'(E)` |
| `param_shift_single` | `[L''(E) + (1/2) L'''(E) g_j] Var(f)/S` | covariance of `grad_out` with the primal-derived difference (`L''`) plus the same Jensen term |
| `param_shift_filter` | `[L''(E) p(1-p)(v1 - v0 - p Delta^2) + (1/2) L'''(E) g_j Var f] / S` (delta method; reps=1, no empty branches) | primal samples reused for both the loss factor and the conditional means; `m_b`, `v_b` = mean and variance of `f` with gate `j` forced to branch `b`, `Delta = m1 - m0`, `g_j = p(1-p) Delta` |

So `_inf` is exactly unbiased only when `L''' = 0` (quadratic loss); the quadratic formulas above are the special case. Measured on the 6-gate circuit, bit 1, 20,000 repeats, bias per gate (the loss `e^3` has `L''' = 6`, `exp(-3e)` has `L''' = -17.465`; the effect concentrates on the last gate, whose `g_j` is large):

| loss | S | `_inf` predicted (last gate) | `_inf` measured (last gate) | `_filter` measured (last gate) |
|---|---|---|---|---|
| `e^3` | 20 | -0.0019 | -0.0019 | -0.0063 |
| `e^3` | 40 | -0.0010 | -0.0009 | -0.0030 |
| `exp(-3e)` | 20 | 0.0056 | 0.0048 (SE 0.0005) | -0.0190 |
| `exp(-3e)` | 40 | 0.0028 | 0.0023 | -0.0091 |

On the other five gates the `_inf` bias is within 0.0007 of zero and the `_filter` bias within 0.0008 (SE up to 0.0005). For nonlinear losses all three methods are biased at `O(1/S)`; `_inf` is the least, `_filter` about three times `_inf` on the gate where it matters on this circuit (in general use the formula), `_single` adds the covariance term `L'' Var/S` everywhere.

**`_filter` formula check (`cookbook_13_bias_theory.py` part 1e, independent keys).** Predicted vs measured bias per gate (`_single` prediction for the sign comparison):

| circuit, loss | S | `_filter` predicted | `_filter` measured | SE | `_single` predicted |
|---|---|---|---|---|---|
| 6-gate mixed, `e^3` | 20 | 0.0003 0.0000 0.0001 0.0001 -0.0002 -0.0058 | 0.0003 0.0000 0.0001 0.0001 -0.0002 -0.0061 | 0.0001 | 0.0055 ... 0.0035 |
| 6-gate mixed, `e^3` | 40 | ... -0.0029 (last gate) | ... -0.0029 | 0.0001 | 0.0028 ... 0.0017 |
| 6-gate mixed, `exp(-3e)` | 20 | ... -0.0201 (last gate) | ... -0.0189 | 0.0005 | 0.0358 ... 0.0417 |
| 6-gate mixed, `exp(-3e)` | 40 | ... -0.0100 (last gate) | ... -0.0099 | 0.0004 | 0.0179 ... 0.0209 |
| 3-gate chain, quad | 20 | 0.0005 0.0004 -0.0043 | 0.0008 0.0004 -0.0043 | 0.0003 | 0.0240 (all) |
| 2-gate, quad | 20 | 0.0031 0.0031 | 0.0028 0.0034 | 0.0003 | 0.0187 (all) |
| 2-gate, quad | 100 | 0.0006 0.0006 | 0.0006 0.0006 (200,000 repeats) | 0.0001 | 0.0037 (all) |

The formula reproduces the gate pattern and the sign. On the 2-gate circuit the `_filter` bias has the **same** sign as `_single`; on the 3-gate chain and the 6-gate last gate it is opposite. At S=100 on the 2-gate circuit a first 40,000-repeat run put gate 1 3 SE above the prediction; with 200,000 repeats it is 0.0006 against 0.0006 (SE 0.0001), so that was a chance deviation. The matching variance formula for a linear readout, `Var_j = p(1-p)[(1-p) v1 + p v0 + p^2 Delta^2]/S`, gives 0.00685 mean sd on the depth-3 chain at S=1000 against measured 0.00694 (`cookbook_13_bias_theory.py` part 1e).

**Unified K-branch and reps form (`cookbook_13_bias_theory.py` part 1f; form contributed by the reviewer, re-run here).** For branch `k` of a gate with K branches, summed over the repetitions `r` in which the gate appears: `bias = sum_r [L'' p_k (v_k - Var f) + (1/2) L''' p_k (m_k - E) Var f]/S`; for K=2 it equals the formula in the table (`v1 - Var f = (1-p)(v1 - v0 - p Delta^2)`). Matching variance for a linear readout: `Var_k = p_k[(1 - 2 p_k) v_k + p_k Var f]/S`.

| case | S | predicted | measured | SE |
|---|---|---|---|---|
| K=3 `PditCycle`+`PditShift`, quad | 20 | -0.0162 0.0080 -0.0002 | -0.0159 0.0080 0.0001 | 0.0005 |
| K=3, quad | 50 | -0.0065 0.0032 -0.0001 | -0.0068 0.0033 -0.0002 | 0.0003 |
| K=3, cubic | 20 | -0.0330 0.0224 0.0079 | -0.0323 0.0164 0.0091 | 0.0019 |
| K=3, cubic | 50 | -0.0132 0.0090 0.0032 | -0.0133 0.0099 0.0027 | 0.0010 |
| reps=2 `PNOT`+`PCNOT`, quad | 20 | 0.0015 0.0009 | 0.0020 0.0006 | 0.0004 |
| reps=2, quad | 50 | 0.0006 0.0004 | 0.0004 0.0001 | 0.0002 |

The cubic S=20 row shows the second-order expansion straining (second component 3 SE off), as for `_single`.

**Validity domain: empty branches (`cookbook_13_bias_theory.py` part 1g).** The delta-method formula assumes every branch receives samples. `sampled.py` computes `p * (E1_hat - E_hat) * L'(E_hat)` per K=2 gate and sets an empty branch's conditional mean to 0. So when branch 1 is empty (probability `(1-p)^S`) it returns `-p E0_hat L'(E0_hat)` (factor `p`, not `p(1-p)`), and when branch 0 is empty (probability `p^S`) `E1_hat = E_hat` and it returns exactly 0. For a quadratic loss and a 0/1 observable the expected estimate needs no expansion: condition on `n1 ~ Binomial(S, p)`; within a branch the samples are iid, so `E[E_b_hat] = m_b` and `E[E_b_hat^2] = m_b^2 + v_b/n_b`, and the sum over `n1 = 0..S` (with the two empty-branch cases above) is exact (`exact_quad` in the script). Measured on the 2-gate circuit (`m0 = 0`) and on a 3-gate circuit `[PNOT(0), PCNOT, PNOT(1) theta=0.5]` (`m0 = 0.622`, `m1 = 0.500`), loss `(E_hat - 0.8)^2`, 40,000 repeats:

| circuit, theta0, S | P(n1=0) | P(n0=0) | delta method | exact binomial sum | measured (SE) |
|---|---|---|---|---|---|
| 2-gate, -3, 20 | 0.378 | 0.000 | 0.00108 | 0.01498 | 0.01505 (0.00014) |
| 2-gate, -3, 100 | 0.008 | 0.000 | 0.00022 | 0.00051 | 0.00059 (0.00009) |
| 2-gate, -2, 20 | 0.079 | 0.000 | 0.00231 | 0.00961 | 0.00980 (0.00028) |
| 3-gate, -3, 20 | 0.378 | 0.000 | 0.00006 | 0.00280 | 0.00280 (0.00004) |
| 3-gate, +3, 20 | 0.000 | 0.378 | 0.00000 | 0.00000 | 0.00003 (0.00008) |
| 3-gate, -2, 20 | 0.079 | 0.000 | 0.00014 | 0.00157 | 0.00160 (0.00009) |

All six rows agree within 1 SE. The `theta0 = +3` row shows that an empty branch 0 adds no bias. An earlier first-order "correction" in this file used the wrong factor (`p(1-p)`) and a spurious branch-0 term; it only looked right because the 2-gate circuit has `m0 = 0`, which hides the factor. For non-quadratic losses there is no closed form: use the delta-method formula only when `S min(p, 1-p) >= 10` (then `P(empty) < 1e-4`), otherwise raise `S` or measure.

**Remedy scope.** The two-batch product `(E_hat_1 - t)(E_hat_2 - t)` removes the bias for a **quadratic** loss only. For a degree-`k` polynomial loss use `k` independent batches (a U-statistic); verified for the cubic `e^3` with `param_shift_inf`, S=40, 20,000 repeats, last-gate bias: single batch -0.0010, two independent batches (`E1 E2 E1`) -0.0003, three independent batches (`E1 E2 E3`) 0.0000 (other gates 0.0000). For non-polynomial losses such as `exp` there is no exact finite-batch fix: raise `S` or accept the `O(1/S)` bias.

**Validity of the second-order formula.** The formulas drop `O(1/S^2)` terms. Ratio measured/predicted for the `_single` bias at S=20, 20,000 repeats: cubic 1.09-1.18 (the last gate 1.18), `exp(-3e)` 0.91-0.99; at `S >= 40` within a few percent (rows above). Use it as an estimate, not an identity, when `S <= 20`.

**Fixed points (`cookbook_13_bias_theory.py` part 2; single `PNOT`, loss `(E_hat - 0.8)^2`, plain SGD lr 0.5, 3000 steps, tail-averaged exact E over the last 1000 steps, mean of 4 seeds).** Setting the expected gradient to zero gives closed forms (quadratic loss, so `L''' = 0` and `_inf` has no bias): `_inf`: `E* = t`; `_single`: `2(E - t)E(1-E) + 2E(1-E)/S = 0`, so `E* = t - 1/S`; `_filter`: the gradient is `2(E_hat - t) p (1 - E_hat)`, so `(E - t)(1 - E) = E(1-E)/S` and `E* = t/(1 - 1/S)`.

| S | method | measured E* | predicted |
|---|---|---|---|
| 20 | `_inf` | 0.8002 | 0.8000 |
| 20 | `_single` | 0.7509 | 0.7500 |
| 20 | `_filter` | 0.8434 | 0.8421 |
| 50 | `_inf` | 0.8006 | 0.8000 |
| 50 | `_single` | 0.7794 | 0.7800 |
| 50 | `_filter` | 0.8176 | 0.8163 |

They agree to 0.0013. (An earlier protocol with lr 0.02 had not converged and sat 0.002-0.010 away; the gap was insufficient convergence, not sigmoid averaging.) For a single gate from a fixed input `_filter` has a bias of the **opposite sign** and similar order to `_single`'s: at S=20 the measured fixed points are 0.8434 (`_filter`) and 0.7509 (`_single`) around the target 0.8. This is not general: the `_filter` sign follows `v1 - v0 - p Delta^2` (formula in the per-method table above) and on the 2-gate circuit of part 1e it has the same sign as `_single`.

**Multi-gate case (3-gate chain, S=20).** With several gates there is no closed form. Deterministic Adam (lr 0.05, as in the sampled run; `cookbook_13_bias_theory.py` part 2) on `true gradient + 2 Var(f)/S` ends at E = 0.7500 after 400 steps with theta = [-18.0248 -18.2815 1.0986], and again at E = 0.7500 after 4000 steps with theta = [-197.0857 -197.0417 1.0986]. So 0.7500 is a **degenerate limit**, not a joint fixed point: the constant bias `2 Var(f)/S` pushes the first two logits to -infinity (their `dE/dtheta` vanishes, so nothing balances it) and the last gate then satisfies the single-gate condition `E* = t - 1/S`. A gate that receives no gradient signal drifts under a constant bias; this is the testable signature. The sampled `_single` single-batch runs (`cookbook_12_study.py` part 1b, 6 seeds) ended at 0.7367 +/- 0.0313 (sd, so SE = 0.0128): about one SE from 0.7500, because 400 noisy Adam steps do not drive the first two logits to -infinity. The same part 1b gave `param_shift_inf` 0.7985 +/- 0.0159, `param_shift_single` with the two-batch product 0.8035 +/- 0.0121 and `param_shift_filter` single batch 0.8167 +/- 0.0127 (mean +/- sd over 6 seeds); for `_filter` the per-gate bias on the chain is predicted by the delta-method formula above (gate 3: -0.0043 at S=20), but its converged value under Adam was not simulated, so the 3.2 SE excess is explained only qualitatively.

**REINFORCE baselines (`cookbook_11` part 3, M=1000, 200 repeats).** No baseline: sd 0.0104; batch mean with the `M/(M-1)` factor: sd 0.0073; leave-one-out baseline (`baseline_i` = mean of the other trajectories, no correction needed): sd 0.0073, same bias (max 0.0007). A baseline cuts the sd by 30% here; LOO is the simpler unbiased choice.

**Training at equal cost (`cookbook_12_study.py` part 1; 7,000 circuit runs per gradient, 150 Adam steps, 6 seeds, same 3-gate task):** with `lr=0.05` all four gradient sources (`_inf` S=1000, `_single` S=1750, `_filter` S=7000, REINFORCE M=7000) end at exact objective about 0.9885 and reach 0.95 at step 55-56; with `lr=0.2` all reach 0.9989 and 0.95 at step 13. On this easy task gradient noise is not the bottleneck, the learning rate is; do not read a ranking into it. A sampled objective has `SE = sqrt(E(1-E)/S)` (0.0020 at `E=0.998`, `S=500`): stop when the moving average changes by less than 2 SE and confirm with the exact objective on a tiny circuit.

**Training loop actually run (`cookbook_11` part 1):** Adam (`optax`, lr 0.1, 300 steps) maximising `E[bit 1]` of the 3-gate chain; the final **exact** objective is 0.9986 with the exact gradient, `param_shift_inf` (S=500), `param_shift_filter` (S=3000) and REINFORCE (M=3000) alike (theta about `[-4.1, -4.4, 6.7]`). What to print when debugging: the exact objective every few steps (never the sampled one alone), the gradient norm against an exact gradient on a tiny circuit, and the sampled/exact ratio.

**The discrete gate has no pathwise gradient (`cookbook_11` part 4).** For the regime circuit (`PditCycle` then `MixtureGaussianGate`) `d E[x] / d theta_cycle` is analytically `[-0.1224, -0.4177]`; `jax.grad` through `HybridPCircuit.sample` returns `[0, 0]` for the `PditCycle` theta (the continuous `means` and `log_vars` parameters do get correct gradients). That makes the total gradient **biased** unless you add the score-function term `x * d log p(branch)/d theta`: 400,000 samples gave `[-0.1223, -0.4171] +/- [0.0011, 0.0010]`.

**Bias and variance of `_filter` (`cookbook_9_variance.py` parts 1b, 4, 7; `cookbook_6_gradients.py`).** It is a ratio estimator and empty branches are set to 0. For a single gate the conditional means are exact and the extra variance comes from using the **empirical** overall mean in place of `p`: `g = p (1 - p_hat)` with `p_hat = Binomial(S, p)/S`, so `sd = p sqrt(p(1-p)/S)` (single `PNOT`, S=2000, 400 repeats: p=0.20 predicted 0.00176, measured 0.00178; p=0.50 0.00559 / 0.00574; p=0.80 0.00715 / 0.00726). For a K-branch gate the general single-gate form is `sd_j = p_j sqrt(Var f / S)` (K=2: `Var f = p(1-p)`; `cookbook_9_variance.py` part 1c, `PditCycle` readout = index, S=2000, 400 repeats: `theta=[0.3, 0.1]` predicted 0.00681, 0.00558 measured 0.00685, 0.00561; `theta=[-1.0, 1.5]` predicted 0.00107, 0.01308 measured 0.00114, 0.01386). On chains the single-gate term (here with `Var f = 1/4`, 0.0079 at `S=1000`) is an approximation, not a bound; the delta-method variance above is the better predictor (depth 3: 0.00685 against measured 0.00694, `cookbook_13_bias_theory.py` part 1e): measured mean sd 0.0069 (depth 3), 0.0080 (depth 6), 0.0077 (depth 10) (`cookbook_9_variance.py` part 8). At `theta=0` the bias is undetectable even at N=20 (0.0057; 1.8 SE). With rare branches it is real, for K=2 and K=3: `theta=-4` (p=0.018), last-gate gradient exact 0.0177, N=20: mean 0.0053 (bias -0.0124, SE 0.0002), N=200: -0.0005, N=2000: about 0. K=3 (`PditCycle`, `theta=[-4,-4]`, loss = E[index], exact `[0.0167, 0.0344]`, 1000 repeats): N=20 bias `[-0.0125, -0.0250]` (SE 0.0002, 0.0005), N=200 `[-0.0004, -0.0008]`, N=2000 about 0. Use `N` large enough that every branch of every gate is sampled many times.

**Silent zeros.** `jax.grad` of a loss built from `sample()` returned `[0.0, 0.0, 0.0]` with no error.

**REINFORCE with baseline works (`cookbook_6_gradients.py`).** Per-gate score-function surrogate in plain JAX (paper Eq. 34; `d log G/d theta = b - p` for a sigmoid gate), baseline = batch mean:

| M trajectories | mean (3 thetas) | sd | exact |
|---|---|---|---|
| 1,000 | -0.0341, -0.0522, 0.1148 | 0.006-0.008 | -0.0373, -0.0524, 0.1153 |
| 10,000 | -0.0367, -0.0529, 0.1143 | 0.002-0.003 | same |
| 100,000 | -0.0373, -0.0525, 0.1151 | 0.0006-0.0008 | same |

Torx does not ship this estimator; the script is the template. Compare it with parameter shift at **equal total circuit runs** (table above), not at equal `S`. The batch-mean baseline introduces an `O(1/M)` bias (`cookbook_9_variance.py` part 3): with `M=20` the uncorrected estimator returned 0.950 of the exact gradient (expected `(M-1)/M = 0.950`) and the `M/(M-1)` correction (as in notebook 09) removes it (gradient bias -0.0063 -> -0.0001). It also works for a joint statistic (any `f(z_k)`), which parameter shift through `expval_all` cannot do.

## Energy-based kernel gradient (paper Eq. 44), run (`cookbook_13_bias_theory.py` part 3)

Gate `G(theta)(y|x) = exp(-E(x,y;theta))/Z(x)` with `E = -(a + b x)(2y - 1)`, `x ~ Bernoulli(0.4)`, readout `f = y`. Exact gradient `[0.4457, 0.1711]`. Eq. 44 score term `f(y) (-dE(x,y)/dtheta + E_{y'~G} dE(x,y')/dtheta)` with the inner expectation done exactly: `[0.4451, 0.1706]` +/- `[0.0008, 0.0007]`; with one independent auxiliary sample `y'` (as the paper prescribes): `[0.4468, 0.1706]` +/- `[0.0013, 0.0009]`, sd per sample `[0.833, 0.559]` against `[0.481, 0.454]` with the exact inner expectation: higher variance, as the paper warns. Both agree with the exact gradient (400,000 samples). With a **downstream gate** (part 3b: the energy-based gate's output is flipped with probability 0.2 and the readout is the flipped bit, so `f(z_k)` multiplies the gate's score) the exact gradient is `[0.2674, 0.1027]`: exact inner expectation (this is REINFORCE with the exact score) `[0.2682, 0.1027]` +/- `[0.0010, 0.0007]`, sd per sample `[0.602, 0.446]`; one auxiliary sample `[0.2692, 0.1032]` +/- `[0.0014, 0.0009]`, sd `[0.904, 0.574]`; a baseline `f - mean f` cuts the auxiliary-sample sd to `[0.611, 0.400]` and the exact-score sd to `[0.389, 0.273]` (means `[0.2670, 0.1033]` and `[0.2676, 0.1032]`). Deeper and at equal cost (part 3c: energy-based gate with a fixed input, so it is also a sigmoid-mixture gate and parameter shift applies, then two flip gates, readout the final bit; exact `dJ/da = -0.0175`; 7,000 circuit runs per gradient, 200 repeats): parameter shift `_inf` (S=1000, times 2) mean -0.0177 sd 0.0104; Eq. 44 exact score without baseline mean -0.0170 sd 0.0084, with baseline `f - mean f` mean -0.0171 sd 0.0055; one auxiliary sample without baseline mean -0.0172 sd 0.0116, with baseline mean -0.0175 sd 0.0080. Budget accounting matters: `_inf`'s 7,000 runs buy **all three** gradients, while Eq. 44 delivers only `d/da`. Parameter shift for `d/da` alone (two deterministic branch circuits, 3,500 samples each) gives mean -0.0176, sd 0.0058 (formula 0.0057): a tie with Eq. 44 exact score plus baseline (sd 0.0055), and better than the auxiliary-sample version with baseline (0.0080), which also draws an extra auxiliary sample per trajectory that is not counted in the 7,000. The exact-score variant needs `E_{y'} dE/dtheta` in closed form, which exists only for a tractable partition function. So for `d/da` alone they tie (parameter shift sd 0.0058 against baseline sd 0.0055). For the **full** 3-parameter gradient at 7,000 runs (`cookbook_13_bias_theory.py` part 3c, 200 repeats; exact `[-0.0088, -0.0069, 0.0118]`): `_inf` (S=1000) sd 0.0053, 0.0052, 0.0054; `_filter` (S=7000) sd 0.0027, 0.0029, 0.0028; REINFORCE with a leave-one-out baseline (M=7000) sd 0.0029, 0.0028, 0.0027. The score-function estimator with a baseline and Torx's own `_filter` have about half the sd of `_inf` at equal runs on this circuit. This is a plain-JAX implementation; Torx has no energy-based gate with this gradient built in.

## What the paper gives (whitepaper Section III C, page 7-8 of the wiki)

- **Objective**: `J(theta) = E_{Y~rho_out}[f(Y)] = <f, G_k ... G_1 rho_in>` (Eq. 32).
- **REINFORCE** (Eq. 34): `dJ/dtheta_j = E[ f(z_k) * d log G_j(z_j | z_{j-1}) / dtheta_j ]`. Unbiased from one batch of trajectories; needs `G_j > 0` on the support (stay away from `p in {0,1}`); sigmoid parametrisation guarantees it. High variance. **Torx does not implement REINFORCE**; you write the surrogate yourself.
- **Parameter-shift** for sigmoid-mixture gates: `dG/dtheta = G(theta) - G(theta^-)` with `sigma(theta^-) = sigma(theta)^2` (Eq. 36-37), estimator variance `(Var_+ f + Var_- f)/S` (Eq. 40). Applies only to gates of form Eq. 17, and needs one shifted circuit per parameter.
- **Energy-based kernels** (Eq. 41-44): score-function style, high variance.

## Writing REINFORCE yourself (notebook 09 pattern and its catch)

Notebook 09 (MaxCut) samples indices from the **exact** `StateVectorSimulator` probability vector and differentiates the exact `log p` of the sampled indices, with the batch mean as baseline and the finite-batch correction `M/(M-1)` applied to the centered surrogate. That needs the full `2^n` density, so it cannot scale and is not hardware-style sampling. A hardware-feasible estimator follows the paper's Eq. 34: run the circuit, record each gate's branch, weight `d log G_j` by `f(z_k) - baseline`. For a sigmoid gate `d log G / d theta = (1 - p)` if the branch was taken, `-p` if not.

## Training loop template

`cookbook_1_core.py` has the gradient check. A training loop with optax (optional extra; `pip install optax`):

```python
opt = optax.adam(1e-2); state = opt.init(thetas)
@jax.jit
def step(thetas, state, key):
    l, g = jax.value_and_grad(loss)(thetas, key)
    upd, state = opt.update(g, state); return optax.apply_updates(thetas, upd), state, l
```

Before trusting a sampled gradient: compare with the exact gradient on a tiny circuit, 20+ repeats, report mean and standard error (pattern of `TestGradientUnbiasedness`).

## Weight tying

`param_map` is **not a Torx API**. It is a numpy index array inside notebook 08: `full_thetas = gate_logits[param_map][:, None]`, so 60 gate positions read 5 logits and gradients accumulate through the indexing. `len(thetas)` must equal the number of gates, so expand before `build_circuit`. Library mechanisms: `Site.param_key` (sites with equal key share one param slice), `TiledFactor(weight_tied=True)`, `ChainFactor(weight_tied=True)`, and `circuit reps`.
