# Estimator math (verified numerically in `examples/cookbook_4_attention.py`)

## p-bit dot product

Independent spins `s_i in {-1,+1}` with `E[s_i] = m_i`. For two blocks `a, b` with means `ma = q/s_q`, `mb = k/s_k` (`s = 1.05 * max|.|` maps into `(-1,1)`):

- `E[s_a,i s_b,i] = ma_i mb_i`, so `q.k = s_q s_k E[sum_i s_a,i s_b,i]`.
- Per-sample variance `sum_i (1 - (ma_i mb_i)^2)`; per-term variance is `1 - m^2` because `(s_a s_b)^2 = 1`.
- **Standard error of the estimate** after `N` independent samples:
  `std = s_q s_k sqrt( sum_i (1 - (ma_i mb_i)^2) / N )`.
- Measured (d=288, N=1000, 200 repeats): formula 1.95, measured 1.81 (about 1.4 SE of the 200-repeat sd).
- Relative error: `std / |q.k|`. With a small `|q.k|` relative to `s_q s_k sqrt(d)` the signal is buried; the dot-product SNR is `|q.k| sqrt(N) / (s_q s_k sqrt(d_eff))`.
- Consequence: 10x accuracy costs 100x samples. Key scores that share the `q` block have error covariance `sq^2 sk^2 sum_i Var(s_q,i) m_kj,i m_kl,i / N / d` (sign depends on the key vectors). Measured by `cookbook_4_attention.py` part 6 (400 repeats, N=300): mean off-diagonal correlation of the score errors -0.011 (min -0.101, max +0.061), i.e. close to zero; do not count on cancellation.

## Softmax of noisy scores

`softmax(z + e)` with `e ~ N(0, sigma^2)` per logit: when logits span O(1), error in `w` grows with `sigma`. With scores `q.k_j/sqrt(d)`, `sigma_logit = std/sqrt(d)`.

Measured (numpy, not Torx or THRML; 8 keys, logits spread over [-1.5, 1.5], max weight 0.36, 20 repeats, mean +/- 95% t-CI with t=2.093). TV here is between two softmax weight vectors, not a histogram, so no multinomial floor applies:

| N | TV(softmax true, softmax estimated) | argmax correct |
|---|---|---|
| 100 | 0.117 +/- 0.0225 | 0.90 |
| 1000 | 0.036 +/- 0.006 | 1.00 |
| 10000 | 0.013 +/- 0.002 | 1.00 |

Required N for a logit std of 1.0: about 13; std 0.3: about 147; std 0.1: about 1,323 (script prints it).

**Do not test attention with a saturated softmax.** The earlier `z1_attention.py` demo used a key with score 14.1 against about 0.7; the weights were `[2e-6, ..., 0.999995]`, so `max|dw| = 0.0000` at every N and nothing was tested.

## Quantised and clipped bias

If a bias DAC has `n` bits over `|theta| <= 3` (`cookbook_4_attention.py` part 4) with `E[s] = tanh(theta)` (THRML/Ising bias convention, `P(s=+1) = sigmoid(2 theta)`; the Torx `PNOT` logit convention is `m = tanh(theta/2)`, see SKILL.md): step `6/(2^n - 1)`. Measured (d=288): max error in a mean `|dm|` is 0.196 at 4 bits, 0.047 at 6 bits and 0.012 at 8 bits; bias in the first logit +0.162 at 4 bits, +0.064 at 6 bits and +0.002 at 8 bits. Clipping means to +/-0.999 is negligible. The DAC width for Z1 is **not known**; the 8-bit figure comes from X0 (hardware-facts.md says what is and is not sourced).

## Total-variation floor

TV of an empirical histogram from an **exact** sampler (`cookbook_4_attention.py` part 5, `cookbook_5_gibbs.py`) with `N` samples over `S` states is not zero, and it **depends on the target**. Measured by simulating `multinomial(N, P)`: uniform over 256 states, N=6000: 0.082 (an upper bound for 256 states), N=60000: 0.026; an 8-site Boltzmann ring (`cookbook_5_gibbs.py`, 256 states, N=6000): 0.047. A sampler TV is evidence of bias only when it is well above the floor computed for **its own** target and N. `cookbook_5_gibbs.py` chromatic Gibbs: TV 0.0457 against a floor of 0.0468, consistent with no bias. Notebook 06 reports 0.0634 (chromatic) and 0.4020 (bond-local PISING); 0.402 is far above any floor, hence real bias. For 0.0634 the floor for nb06's actual target and chain count was not recomputed here, so do not call it zero bias or bias.

## THRML-emulated attention with a bias DAC (`cookbook_8_attention_thrml.py`, part A)

Same non-saturated 8-key problem, 2,592 independent THRML spins, N=10,000, 6 repeats, mean TV +/- 95% t-CI (t=2.571): 4-bit 0.059 +/- 0.006, 6-bit 0.021 +/- 0.011, 8-bit 0.009 +/- 0.004, 16-bit 0.012 +/- 0.003 (weight-vector TV, no histogram floor). The 8-bit and 16-bit rows overlap, so they are not distinguishable at 6 repeats: at N=10,000 the sampling noise (TV about 0.012, as in the numpy table) dominates once the DAC is at least 8 bits; 4 bits is clearly worse. This is an emulation with exact biases and independent samples; real X0 pbits add about 5% probability error [WP p9].

## Correlated samples (`cookbook_8`, part C)

If consecutive samples have lag-1 correlation `rho` (geometric decay, `rho = exp(-dt/tau0)` for a device with relaxation time `tau0`), then `Var(mean) = Var_iid (1+rho)/(1-rho)` and `N_eff = N (1-rho)/(1+rho)`. Simulated check (m=0.3, N=2000, 400 repeats): `rho=0.5` measured 1.32e-3 vs formula 1.36e-3; `rho=0.9` 8.45e-3 vs 8.65e-3. With `tau0 = 100 ns` [WP p20], sampling every 300 ns gives `N_eff/N` = 0.91, every 100 ns 0.46, every 30 ns 0.15 and every 10 ns 0.05. Fast sampling does not buy independent samples; the dot-product standard error uses `N_eff`, not `N`.

## One-hot softmax with spins (key-index sampling), `cookbook_8` part B

Bits `x_k` with `E(x) = -sum_k theta_k x_k + lam (sum_k x_k - 1)^2` have **exactly** `softmax(theta)` on the one-hot states for every `lam`. In THRML spin form `E = -(sum b s + sum J s s)` use `b_k = theta_k/2 + lam (2-K)/2` and `J = -lam/2` on all pairs (degree `K-1`, so `K <= 17` fits a degree-16 graph if the embedding allows; Z1's actual embedding is unknown). Exact enumeration (K=8, theta in [-1.5, 1.5]): mass on one-hot states 0.331 at `lam` = 1, 0.600 at 2, 0.918 at 4 and 0.988 at 6, and TV(conditional on one-hot, softmax) about 1e-15 (the identity). THRML block Gibbs (20,000 samples, one block per spin): `lam=2`: one-hot fraction 0.600, TV 0.008, 21% of consecutive samples identical; `lam=4`: 0.915, TV 0.019, 76% identical; `lam=6`: 0.989, TV 0.060, 97% identical. So `lam` trades leakage off the one-hot subspace against mixing; at `lam=6` the chain is nearly frozen and TV rises because the effective sample count collapses. Nothing here uses hardware.

## End-to-end key-index sampling (THRML emulation, `cookbook_10_key_sampling.py`)

Host computes the logits `z_j = q.k_j/sqrt(d)` digitally and programs `b_k = z_k/2 + lam(2-K)/2`, `J = -lam/2` (K=8, `lam=3`, all-to-all, degree 7). The sampler returns an index (the single up-spin); non-one-hot states are discarded. 20,000 recorded samples, **mean +/- sd over 12 seeds**, `rho` and `N_eff` computed on the **index series** (indicator of each key, averaged):

| Gibbs steps per recorded sample | leakage | TV(empirical, softmax) | mean multinomial TV floor | lag-1 rho | N_eff |
|---|---|---|---|---|---|
| 1 | 0.197 | 0.0139 +/- 0.0047 | 0.0073 | 0.527 | 4,975 |
| 5 | 0.197 | 0.0079 +/- 0.0028 | 0.0073 | 0.040 | 14,848 |
| 20 | 0.197 | 0.0062 +/- 0.0022 | 0.0074 | -0.000 | 16,016 |

With 5 steps per sample the mean TV (0.0079) is within one sd of the floor (0.0073); with 20 it is at it: consistent with an exact softmax sampler once thinned. One step per sample is visibly worse (correlated samples: lag-1 rho 0.527 and N_eff 4975, against about 16,000 one-hot samples).

Bias DAC covering `|b| <= 10` (the offset `lam(K-2)/2` makes the range much larger than the logits), 5 steps per sample, 12 seeds: 4-bit TV 0.2451 +/- 0.0022, 6-bit 0.0585 +/- 0.0034, 8-bit 0.0170 +/- 0.0035, ideal 0.0067 +/- 0.0024 (floor 0.0073). The bias **range** sets the logit resolution as much as the bit count: the bias step is `20/(2^n - 1)` = 0.078 at 8 bits, and because the bias carries `z/2` the step in logit units is **0.157**, against logit gaps of 0.43. This does not remove the digital dot products; it only samples the index.

## Mapping attention onto notebook 14's softmax (the key-norm term)

Notebook 14: `p(h = e_k | v) = softmax_k(theta_k(v))` with
`theta_k(v) = log pi_k - 0.5 sum_i (v_i - mu_ki)^2/sigma_ki^2 - 0.5 sum_i log sigma_ki^2`.
Expanding with shared `sigma`: `theta_k(v) = v.mu_k / sigma^2 - ||mu_k||^2/(2 sigma^2) + log pi_k + (terms independent of k)`.
Identify `v = q`, `mu_k = k_j`. To recover `softmax(q.k_j / sqrt(d))` you need `sigma^2 = sqrt(d)` **and** a bias per key that cancels the norm: `log pi_j = ||k_j||^2/(2 sqrt(d))`. A plain `c_k + sum v_i W_ik` hides this: `c_k` is not the mixture weight. The host must compute one number per key. That Gaussian-mixture mapping is a derivation, not run. What has been run (THRML emulation, not hardware) is the direct one-hot Ising sampler for a key index, in the next-to-last section and `cookbook_10_key_sampling.py`.
