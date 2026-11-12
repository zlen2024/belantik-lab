---
name: ebm-training
description: Scientist-level guidance for designing, training, debugging and evaluating Energy-Based Models (EBMs) - Boltzmann machines/RBMs/Ising, deep continuous EBMs, latent-variable and JEPA-style models. Covers maximum likelihood with MCMC (CD, PCD, replay buffers), score matching (SM/DSM/SSM), noise-contrastive estimation, Stein and adversarial training, plus LeCun's energy/loss-functional/collapse framework. Use whenever the user trains, compares or debugs an EBM, asks which EBM training method to use, sees diverging energies or mode-weight errors, or wants to train on THRML/Torx/Extropic hardware. Use together with the `thrml` skill (Ising/block-Gibbs code) and the `torx` skill (stochastic circuits as samplers).
---

# EBM training

Base: Song & Kingma, *How to Train Your Energy-Based Models* (arXiv:2101.03288v2), plus LeCun et al. 2006 (energy-based learning), LeCun 2022 (JEPA), and the sources in `references/sources.md`. Saved research notes default to Bahasa Melayu with English technical terms; this file stays English because it is agent instructions.

## 0. How to think (the LeCun frame)

An EBM does not model a probability. It models a **scalar compatibility** `E(x)` (or `E(x,y,z)`); lower means more compatible. Probability is optional: `p(x) = exp(-E)/Z` is one way to read an energy, and the intractable `Z` is the whole difficulty. Keep three things separate, because most confused EBM work mixes them:

1. **Architecture** - what function `E_theta` is (MLP, CNN, Ising `-sum J s s`, RBM free energy, JEPA predictor distance).
2. **Inference** - how you use the energy: `argmin_y E(x,y)`, sampling `p ~ exp(-E)`, or latent minimisation/marginalisation `F(x) = min_z E(x,z)` or `-log sum_z exp(-E)`.
3. **Loss functional** - how `theta` is shaped. Every working loss pushes energy **down on data** and **up somewhere else**. A loss that only pushes down collapses to a flat energy. Methods differ only in *where* they push up and *what* stops collapse (`references/lecun-framework.md`).

Collapse-prevention has five families; name which one your method uses before running anything:

| Family | Pushes up on | Methods | Scales badly when |
|---|---|---|---|
| Contrastive, model negatives | samples of the current model | MLE+MCMC, CD, PCD, replay buffer | chains do not mix (multimodal, high dim) |
| Contrastive, fixed negatives | a known noise density | NCE | noise far from data in high dim |
| Local / score based | nothing global; matches `grad_x log p` | SM, DSM, SSM, Stein | low-density gaps between modes (blind to mode weights) |
| Regularised / architectural | nothing; limits low-energy volume | JEPA+VICReg, bottlenecks, sparse latents | needs good regulariser; no normalised density |
| Learned negatives | a generator `q_phi` | adversarial / variational (Eq. 31-32) | max-min instability; `q` expressivity |

## 1. Choose the method (decide in this order)

1. **Do you need a normalised density / likelihood, or only samples or a score?** Only samples or scores -> skip `Z`.
2. **Discrete spins/bits?** -> MLE with block Gibbs (THRML), exact for the Gibbs step; SM does not apply directly (it needs continuous `x`); discrete options (ratio matching, Gibbs-with-gradients, energy discrepancy) are in `references/advanced-topics.md` section 1.
3. **Continuous, d small (<= ~10)?** -> NCE or DSM first (cheap), verify mode weights with an exact grid. NCE needs noise that covers the data (our sigma_noise 0.8 failed, KL 4.1); single-sigma DSM weights are noisy (0.184 +- 0.033).
4. **Continuous, high d?** -> DSM with noise levels (score-based) or MLE/CD with a replay buffer, gradient clipping, correct Langevin noise and a Lipschitz energy. Energy L2 regularisers and buffer restarts are common advice but **hurt in our 2-D toy only** (`results.md` R3; high-d behaviour untested): treat them as hyper-parameters to ablate, not defaults (`references/training-recipes.md`).
5. **Need the energy itself to be calibrated across modes?** Use NCE (with a noise scale that covers the data) or MLE with a sampler whose mixing you have verified (multi-chain R-hat/ESS, section 2 rung 4); do **not** rely on SSM alone ([own, 3 seeds]: `references/results.md`).
6. **No good negatives/high-dim, representation learning goal?** Regularised JEPA-style (no `Z`), not a likelihood.
7. **Hardware (Z1/XTR-0) target?** The model must be an Ising/Boltzmann machine; see `references/thrml-torx-mapping.md` for degree and graph constraints.

## 2. The verification ladder (never skip a rung)

EBM code fails silently: a wrong sign or a biased chain still produces plausible curves. Before any real run:

1. **Gradient identity** on a model small enough to enumerate: the sampled contrast equals autodiff of the exact KL (`scripts/gradient_identity_check.py`, under a minute; mean of 5 keys within 4 standard errors of the exact gradient). Its `grad log Z` line only checks your enumeration conventions; the THRML-vs-exact line is the real test. For score matching run `scripts/score_matching_sign_check.py` first.
2. **Exact reference**: for <= ~24 spins enumerate all states and report `KL(p_data || p_model)` exactly. Never report only reconstruction error or a loss value - both can improve while the model gets worse.
3. **Known-answer toy** with a property you can name (e.g. unequal mode weights) before images. `scripts/continuous_ebm_zoo.py`.
4. **Sampler audit**: R-hat across dispersed starts, ESS, TV against the exact law with its iid floor (`scripts/mcmc_mixing_diagnostics.py`), and short-run vs long-run statistics (Nijkamp et al. 2019). If they differ, your model is a *generator trained with that sampler*, not the density you wrote. Use the multi-chain ESS (includes between-chain variance) and R-hat together; a within-chain ESS hides trapped chains ([own], `results.md` R6).
5. **Only now** scale up, one change at a time, with at least 3 seeds. State what was measured exactly and what was inferred.

## 3. Method cheat-sheet (equations from the paper; fixes noted)

- **MLE gradient** (Eq. 3-4): `grad_theta log p(x) = -grad E(x) + E_{x'~p}[grad E(x')]`. Needs samples from `p`.
- **Langevin** (Eq. 6): `x <- x - (eps^2/2) grad_x E + eps z`. The `noise = eps` coupling is what makes it a sampler; shrinking the noise (common in image EBMs) trades correctness for speed and creates mode-seeking chains.
- **CD / PCD / buffer** (Sec. 3): chains from data, persistent, or replayed. Biased, but how much is empirical: [own, 3 seeds] on a 14-spin RBM the CD-1 gradient has excess bias 0.56 (cosine 0.82) yet training reaches KL 0.05; CD-10 is nearly unbiased; 1-sweep PCD (128 chains, lr 0.02) was worst (0.64 +- 0.36) although its fixed-theta gradient excess is only 0.14 (chain lag, hypothesis). Always compare with an exact reference.
- **Score matching** (Eq. 8): `E_data[0.5 ||grad E||^2 - tr(Hess E)]`.
  **Errata:** the arXiv v2 text prints `+ d2E/dx2` in Eq. 8, 15 and 17 (and in the Taylor form of section 5.1). With `p ~ exp(-E)` the correct sign on the Hessian term is **minus**. Verified: 1D Gaussian recovers the precision only with `-` (`results.md`), and SSM diverged in our run with `+`. Treat the printed sign as a trap whenever you implement from the paper.
- **DSM** (Eq. 10): `0.5 ||grad log q(x~|x) - grad log p_theta(x~)||^2`. Inconsistent unless noise -> 0, high variance as sigma -> 0 (control variate Eq. 13-14).
- **SSM** (Eq. 17): sliced; use a Hessian-vector product (`jax.jvp` of `grad`), cost linear in d.
- **NCE** (Eq. 18-21): logistic regression data vs noise with `log Z` a learnable scalar; gives `Z` as a by-product. Noise close to, but not equal to, data.
- **Stein / KSD** (Eq. 30), **adversarial** (Eq. 31-32): see `references/theory-map.md`.
- **Free energy for latent models:** `F(v) = -log sum_h exp(-E(v,h))`; RBM gradient = data-clamped minus free moments, which is what THRML's `estimate_kl_grad` computes (`ΔW = -β(<ss>+ - <ss>-)`).

## 4. Failure triage (symptom -> likely cause -> test)

| Symptom | Cause | Test / fix |
|---|---|---|
| Energy of data and negatives drift to large negative values; loss unbounded below | CD objective has no bound; energy scale free | clip gradients, check the Langevin noise term, try a small `reg * mean(E_pos^2 + E_neg^2)` (Du & Mordatch 2019) **and measure per-mode mass**: in our toy reg 0.01 already cost 10x KL; monitor `E_data - E_neg` |
| THRML CD trains worse than expected | chain init: THRML samples free blocks in **list order**, so `[vis, lat]` with `init=[data, h]` resamples vis from h and discards the data (CD becomes a short run from noise); CD-1 is also genuinely biased (excess 0.56, cosine 0.82) | initialise `h ~ p(h|data)` (or list latents first); compare the mean gradient with the exact one (`thrml_rbm_negative_phase.py`). A bug here once made CD-1 look divergent (KL 2.1 -> 18) in an earlier version of this skill |
| Correct modes, wrong mode weights | (a) score losses see little of the valley carrying the weight (SSM 0.126 vs 0.200); (b) over-regularised energy (cd_k reg 0.2: 0.48); (c) chains that cannot cross barriers | check mass per mode against an exact grid; NCE with broad noise, lower regularisation, tempering (`advanced-topics.md`) |
| Replay buffer collapses to one mode | observed: restart 0.05 -> small-mode mass 0.001, restart 0 -> 0.218 (true 0.200). Mechanism **untested** (consistent with: noise-restarted chains too short to reach the modes - mass 0.001 / 0.007 / 0.042 at 30 / 100 / 300 steps; 5 steps at restart 0 gives 0.599) | try restart 0, data-like restarts or more Langevin steps; always report per-mode mass |
| SM/SSM gives spurious deep minima outside data | no constraint where there is no data | add noise (DSM), bound/regularise energy, check argmin of `E` over a grid |
| SSM blows up | wrong Hessian sign (see errata) | `scripts/score_matching_sign_check.py` (runs the real loss): recovered precision must be `1/sigma^2` |
| Adversarial `q` lags, student diverges | inner loop too short or `theta` step too large | more inner steps, smaller `theta` lr (`results.md`: lr 0.05 diverged, 0.01 stable) |
| Sampled gradient unbiased in theory but wrong | state dtype, init shapes, key reuse, wrong clamp blocks | gradient identity script; THRML `init_pos` must not be `[]` |
| Good samples, bad likelihood (or vice versa) | short-run MCMC trained as a generator | report which; do not compute likelihood from the energy |

## 5. Reporting standard (what a reviewer would demand)

- Name the loss functional, the collapse-prevention mechanism, and the sampler with its length.
- Give an **exact** reference whenever the model is small enough; otherwise a forward/reverse AIS bracket (`scripts/ais_logz.py`; Salakhutdinov & Murray 2008, Burda et al. 2015) or a stated proxy.
- Report seeds, confidence, and what was *not* tested. Distinguish author claims (paper), your interpretation, and your own measurements (tag them).
- Learned preprocessing stays inside training folds; test data is frozen. See `ml-research-paper-reading`.
- One variable per ablation. If two methods differ in sampler length AND objective, you have learned nothing.

## 6. Using THRML, Torx and Extropic here

- **THRML** (`/thrml` skill): MLE for Ising/RBM with `estimate_kl_grad`; custom negative-phase initialisation gives CD/PCD; run block Gibbs for the sampler audit.
- **Torx** (`/torx` skill): a `DiscretePCircuit` + `StateVectorSimulator.density` gives an *exact, differentiable* `q_phi(x)` with exact entropy for small `n`, the ingredient the adversarial objective (Eq. 31-32) needs; also the single-site conditional Gibbs kernel (`PReset` + `PNOT`). Run Python from outside `master/`.
- **Extropic cloud** (`/extropic` skill): heavy training submits through `extro-sim`; cost and one-job-at-a-time caveats are in memory notes.
- `Z` beyond enumeration: `scripts/ais_logz.py` (forward + reverse AIS bracket checked against the exact value).
- Hardware-aware evaluation: `scripts/hardware_robustness.py` (quantisation and calibration noise); tempering and mixing: `scripts/mcmc_mixing_diagnostics.py`.
- Details, working patterns and pitfalls: `references/thrml-torx-mapping.md`, `references/advanced-topics.md`.

## 7. Where to look

| Need | File |
|---|---|
| Every equation in the paper, relationships, errata | `references/theory-map.md` |
| LeCun framework: energy, loss functionals, collapse, latent variables, JEPA | `references/lecun-framework.md` |
| Recipes: stability tricks, RBM practice, diagnostics, Z estimation | `references/training-recipes.md` |
| THRML/Torx patterns, constraints | `references/thrml-torx-mapping.md` |
| Papers, repos (with verification status) | `references/sources.md` |
| Numbers from our own runs, with commands | `references/results.md` |
| Discrete EBMs, mixing/tempering, Z bounds, composition, hardware training | `references/advanced-topics.md` |
| Runnable checks | `scripts/` (run from outside `master/`: `cd /tmp && python -I <abs path>`) |
