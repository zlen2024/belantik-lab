# Training recipes and diagnostics

Tags: [P] Song & Kingma PDF; [own] our runs (`results.md`); [recall] from memory, verify before relying on it for a paper.

## 1. Discrete / Boltzmann machines (THRML)

- **Spins vs bits.** THRML uses spins {-1,+1} stored as bool (`True` = +1). Binary RBM literature uses {0,1}; mapping `s = 2v - 1` rescales and mixes the parameters: `J_s = J_v / 4`, `b_s = b_v / 2 + Σ_j J_v / 4` (per unit, summing over its neighbours), plus a constant. Do not copy hyper-parameters across conventions without converting.
- **Gradient is a contrast of moments** (`ΔW = -β(<ss>+ - <ss>-)`). [own] verified against the exact KL (`gradient_identity_check.py`).
- **Initialisation** [recall, Hinton 2012]: small weights, visible biases `log(p/(1-p))` of the data mean; THRML's `hinton_init` initialises chains by the marginal bias.
- **Negative phase budget** [own, 3 seeds, `results.md` R4]: final exact KL CD-1 0.049, CD-10 0.052, 100-sweep 0.025, 1-sweep PCD 0.64 (lr 0.02). Gradient excess bias (2048 chains each): CD-1 0.56, CD-10 0.05, PCD 0.14, long consistent with 0 down to the noise floor of about 0.10. Chain initialisation order matters in THRML (`thrml-torx-mapping.md`). Increase k as learning proceeds (growing CD); PCD needs a learning rate small enough that the chains track the model (untested here).
- **Monitoring** [recall, Hinton 2012]: reconstruction error is a poor proxy for likelihood. Use (a) exact KL when n is small; (b) the **free-energy gap** `mean F(v_train) - mean F(v_val)` for overfitting; (c) AIS for `log Z` otherwise (Salakhutdinov & Murray 2008). Closed-form spin-RBM free energy: `F(v) = -b_v·v - Σ_j log(2 cosh(b_j + Σ_i J_ij v_i))` [own, verified to 6e-7].
- **Regularisation** [recall]: weight decay on `J`; sparsity targets for hiddens; mini-batch 10-100 for CD.
- **Hardware constraint**: Z1 is a hardwired sparse degree-16 graph [WP p9 Fig. 7, via `torx/references/hardware-facts.md`]; see `advanced-topics.md` section 7.

## 2. Continuous deep EBMs with MCMC (CD / PCD / buffer)

The set of tricks that made image EBMs trainable (Du & Mordatch 2019; Nijkamp 2019; Grathwohl 2020b) [recall]:

1. **L2 penalty on energies** `reg·(E_pos² + E_neg²)`: recommended in the literature to fix the free energy scale [recall]. [own, 3 seeds] in the 2-D toy it **hurt**: cd_k KL 0.015 / 0.153 / 0.680 for reg 0 / 0.01 / 0.2, small-mode mass 0.186 / 0.459 / 0.483 vs true 0.200. A stable run needed correct Langevin noise and gradient clipping, not the penalty. Use the smallest value that keeps energies bounded and verify per-mode mass.
2. **Replay buffer** (Du & Mordatch recommend ~5% restarts from noise) [recall]. [own, 3 seeds] uniform-noise restarts **collapsed** our buffer: small-mode mass 0.218 / 0.001 / 0.000 and KL 0.009 / 0.986 / 2.171 for restart 0 / 0.05 / 0.2. Setup-dependent: check per-mode mass and consider restarting from data-like points.
3. **Spectral norm / smooth activations** for a Lipschitz energy; softplus/SiLU rather than ReLU when you need second derivatives (SM, SSM).
4. **Decoupled Langevin noise**: smaller noise than `eps` speeds mixing visually but removes the sampling guarantee (Eq. 6): you then train a short-run generator (Nijkamp 2019), not a density. If you use it, say so.
5. **Gradient clipping** and a cosine learning-rate decay. [own] used in all zoo runs.
6. **Data noise** (small Gaussian) stabilises training when data are discrete/bounded [P §4.1 reasoning].
7. **Improved CD** (Du et al. 2020): add the term that backpropagates through the sampler/entropy correction; data augmentation and multi-scale processing [recall].

## 3. Score-based training

- Use **DSM with several noise levels** (NCSN): sigma geometric from large to small, loss weighted by sigma² [P §4.4; Song & Ermon 2019].
- For single-scale DSM choose sigma ≈ data length-scale / 5; too small -> variance ~ 1/sigma² [P Eq 12], too big -> inconsistent [P §4.1]. [own, 3 seeds] sigma 0.3 on sigma_data 0.45 gave small-mode mass 0.184 ± 0.033 (true 0.200).
- **SSM sign**: minus on the Hessian term for an energy (`scripts/score_matching_sign_check.py`).
- After training, **sample with annealed Langevin / reverse diffusion**, not a single-noise-level chain.
- Always check that the score is correct *outside* the data: `argmin E` over a grid; SM has no force there. (In an early debugging run a wrong-sign SSM left a far deeper spurious minimum than at the data; not reproducible from the shipped scripts, so no number is quoted.)

## 4. NCE

- Noise broad enough to cover the data, narrow enough to be informative: [own, 3 seeds] isotropic Gaussian noise for modes at ±2 (data sigma 0.45): noise sigma 0.8 failed (KL 4.1), 1.5 was best (KL 0.0132, mass 0.199), 2.5 and 4.0 were slightly worse (0.0156, 0.0213).
- Make `log Z` a free scalar parameter; in the optimum it matches the true value [own, `results.md` R3 log Z line: matches at sigma_noise 1.5-4.0, wrong at 0.8]. Fixing `Z=1` gives a self-normalised model but needs a very expressive `E` [P §5].
- Use ν = ratio noise:data (here 1). Balanced classes give the cleanest signal.
- Adaptive noise (flow contrastive estimation) for high dimension [P §5].

## 5. Latent-variable EBMs

- Free energy `F(x) = -log Σ_z exp(-E(x,z))` is exactly what enters the MLE gradient: `E_{z|x}[∇E]` (positive) - `E_{x,z~model}[∇E]` (negative). THRML clamps data and Gibbs-samples the latents for the positive phase.
- For continuous latents, amortise inference with an encoder (variational) but keep an exact check on a toy; an amortised inference gap silently biases the positive phase.

## 6. Estimating the partition function

| Case | Method |
|---|---|
| n ≤ ~24 spins | exact enumeration (`scripts/`) |
| RBM | AIS with the bipartite structure (Salakhutdinov & Murray 2008); `scripts/ais_logz.py` brackets it with reverse AIS and checks against enumeration |
| general continuous | AIS (Neal 2001), bridge sampling, or NCE's learned `Z` |
| need only relative comparisons | compare held-out *unnormalised* statistics only if `Z` is shared; otherwise do not |

## 7. Debug order when a run misbehaves

1. Re-run `gradient_identity_check.py` (convention drift).
2. Shrink to the enumerable case and compare KL (`thrml_rbm_negative_phase.py` template).
3. Swap the sampler for a long one; if it fixes the problem the sampler was the bug.
4. Look at `E_data - E_neg`, energy range, per-mode mass, and `argmin E` over a grid.
5. Change one thing; keep 3 seeds; log the exact metric.

## 8. Experiment card (fill before running)

```
Question:
Model E_theta(x):                 Latents / inference:
Loss functional:                  Collapse control:
Sampler (type, steps, noise):     Initialisation:
Exact reference available? (yes/no, how):
Metrics (exact / proxy):          Seeds:
Expected failure mode and the check that would reveal it:
```
