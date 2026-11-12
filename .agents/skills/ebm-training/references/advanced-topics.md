# Advanced topics: discrete EBMs, mixing, tempering, Z bounds, composition, hardware

Evidence tags: [own] our run (`results.md`); [PDF-hw] from `torx/references/hardware-facts.md` (which carries its own source tags); [web] existence confirmed by search this session; [recall] from memory, verify before citing.

## 1. Discrete data (spins, bits, categorical)

Score matching needs `∇_x`, so it does not apply to discrete `x` directly. Options, in the order I would try them:

1. **MLE with block Gibbs** (THRML). Exact conditionals for Ising/RBM; the only approximation is chain length (`thrml_rbm_negative_phase.py`).
2. **Ratio matching** (Hyvärinen 2007) [recall]: match `p(x)/p(x with bit i flipped)`, where `Z` cancels; the discrete analogue of score matching, cost `O(d)` energy evaluations per example.
3. **Gradient-informed samplers** for deep discrete EBMs: *Gibbs With Gradients* (Grathwohl et al. 2021, arXiv:2102.04509 [web]) proposes single-flip moves with probability ∝ softmax of a first-order Taylor estimate of the energy change, then a Metropolis–Hastings correction [recall for the exact form]; discrete Langevin-style samplers (DULA/DMALA, Zhang et al. 2022) [recall].
4. **Energy discrepancy** (Schröder et al. 2023, arXiv:2307.06431 and 2307.07595 [web]): a score-independent, sampling-free loss, with a discrete version. Read before inventing a new discrete objective.
5. **NCE** with a discrete noise model (product of Bernoullis, a mixture, or a learned autoregressive noise).

Convention trap: THRML uses spins `s ∈ {-1,+1}`; binary RBM papers use `v ∈ {0,1}`. `s = 2v - 1` gives `J_s = J_v / 4`, `b_s = b_v / 2 + Σ J_v / 4`; port hyper-parameters (learning rate, sparsity targets) only after converting. **Centering** (Montavon & Müller 2012; Melchior et al. 2016) [recall] reparametrises with mean-subtracted units and makes the {0,1} vs ±1 choice immaterial for optimisation; consider it for deep Boltzmann machines.

## 2. Mixing diagnostics (do not skip)

`scripts/mcmc_mixing_diagnostics.py` runs these on a model with an exact law:

- **Split R-hat** across chains started from dispersed points: values well above 1.05 mean the chains see different regions.
- **Multi-chain effective sample size** (BDA3 / Vehtari et al. 2021: autocorrelation computed from the pooled variance that includes between-chain variance, Geyer pairs). A within-chain estimator centres each chain on its own mean and removes exactly the signal that shows trapping.
- **TV to the exact visible law next to the iid floor** (multinomial of the same size): without the floor a TV number is uninterpretable.
- **Lesson [own]:** a within-chain ESS (my first implementation) reported IAT 2.0 / ESS 1563 at 100 rounds and IAT 68 / ESS 141 at 300 rounds for plain Gibbs while R-hat was 9.0 and 6.3: it was the wrong estimator. The multi-chain ESS is stable (about 17 of 9600 at 300 rounds). Report multi-chain ESS and R-hat from dispersed starts.
- For CD-trained models also compare short-run vs long-run statistics (Nijkamp 2019): the discrepancy *is* the bias of the learned model.

## 3. Parallel tempering (replica exchange)

Model `p_β(x) ∝ exp(-β E(x))`; keep replicas at a ladder `β_0 = 1 > β_1 > …`. After k sweeps per replica, propose swapping neighbours with acceptance
`min(1, exp((β_i − β_j)(E_i − E_j)))` (E without β), alternating even/odd pairs [derivation in the script header]. THRML: `beta` is an `IsingEBM` argument, so build one program per rung and swap states between them. Tune the ladder so neighbour acceptance is roughly 0.2–0.8. [own] On a strongly coupled 8+6 RBM PT cut R-hat from 6.3 to 1.04 and TV to the exact law from 0.086 to 0.020 (iid floor 0.007), whereas plain Gibbs given the same 4x sweeps stayed trapped (R-hat 3.4): the gain is not just more compute (`results.md` R6, 3 chain seeds). TV of a non-mixing sampler is not monotone in run length (it counts which basins the chains landed in), so R-hat and multi-chain ESS carry the conclusion, not TV. Ladder note: the top-pair swap acceptance was 0.90, above the 0.2-0.8 rule, so the top rung could be spaced wider. Use PT for the negative phase of RBMs with deep basins; the cost multiplier is real.

## 4. Bounding `log Z` honestly

- **AIS** (Neal 2001; Salakhutdinov & Murray 2008): `E[Ẑ] = Z`, so `log Ẑ` is a stochastic lower bound on `log Z`; using it makes held-out log-likelihood **optimistic** [recall].
- **Reverse AIS from exact target samples** gives a stochastic **upper** bound on `log Z` (`E[1/Ẑ] = 1/Z`). Using it needs an exact (or verified) sample of the target, which exists only for enumerable models: this forward/reverse sandwich is **bidirectional Monte Carlo** (Grosse, Ghahramani & Adams 2015) [recall], used to validate estimators on small models. **RAISE** (Burda, Grosse & Salakhutdinov 2015) [recall] is the practical variant for real models: it anneals in reverse from the held-out data point (with an approximate posterior over latents) and gives a conservative estimate of `log p(v_test)`. Reporting a bracket still requires stating which of the two assumptions holds.
- Report the number of temperatures, chains, and the spread of `log Ẑ` across runs. A single AIS number with no spread is not a result.
- `scripts/ais_logz.py` [own, `results.md` R8] implements forward AIS and the exact-sample reverse AIS (BDMC-style, validation only) on a 14-spin RBM and checks the bracket against the exact `log Z`; it does not implement RAISE.
- Small models: enumerate.

## 5. Composition and the diffusion link

- **Product of experts**: `E = Σ_k E_k` gives `p ∝ Π p_k`; Hinton (2002) CD was introduced for exactly this. Compositional generation with EBMs (Du et al. 2020, NeurIPS; Du et al. 2023 "Reduce, Reuse, Recycle", ICML) [recall] samples from sums/negations of learned energies with MCMC. Sampling from a *sum* of independently trained energies needs a sampler that mixes under the combined energy; check that with section 2.
- **Score = energy gradient** [P Eq 5]. Diffusion models are score models trained by DSM at many noise scales [P §4.4]; an EBM is the special case where the score is a conservative field `-∇E`. Whether to parametrise the energy or the score directly is an open modelling choice (Salimans & Ho, "Should EBMs model the energy or the score?" [recall, not found by search]): the energy gives a density and composition, the free-form score is easier to optimise.
- **Hardware route** (Extropic DTM, arXiv:2510.23972 [web]): a chain of simple, fast-mixing Boltzmann machines (a denoising sequence) replaces one monolithic EBM, trading expressivity against mixing time. The same trade-off is why section 2 matters.

## 6. Evaluating continuous EBMs beyond a 2D grid

| Goal | Tool | Caveat |
|---|---|---|
| Density quality | held-out log-likelihood via AIS/RAISE bracket | needs a mixing sampler |
| Compare score-trained models | held-out Fisher-divergence (SM) objective, KSD | blind to mode weights |
| Sample quality | FID/IS-type metrics, precision/recall of samples | says nothing about the density |
| OOD / calibration | AUROC of `-E` or free energy (JEM-style) | energy scale is arbitrary across models |
| Mode coverage | classifier counts per mode, 2D marginal grids, comparison with a long run | short-run models can look fine here yet be wrong densities |
| Debugging | plant a known answer (mixture with unequal weights) before real data | what `continuous_ebm_zoo.py` does |

## 7. Training for hardware (THRML/Torx/Extropic)

Facts below carry the tags from `torx/references/hardware-facts.md`; unknowns stay unknown.

- **Graph**: Z1 is a hardwired sparse degree-16 graph [WP p9 Fig. 7]; the logical model must be compiled to 2-local Glauber generators [WP p10]. Enforce the degree in code (`check_degree` in `attn_thrml/ebm_attn_baby`).
- **Bias/coupling precision**: X0 uses 8-bit DACs [WP p20]; Z1 DAC resolution is **unknown**. Sensitivity of a small trained RBM [own, `hardware_robustness.py`, mid-tread quantiser, shared range R, [ASSUME] uniform in parameter space]: exact KL change vs the 0.029 baseline: 5-8 bits within ±0.007 (6 bits even improved by 0.006: the baseline is not at an exact optimum), 4 bits +0.09, 3 bits +0.22.
- **Calibration error**: about 5% absolute probability per site [WP p9]; at p=0.5 that is `δb ≈ 0.1`. [own] On the same RBM `δb = 0.1` raised KL from 0.029 to 0.063, `δb = 0.2` to 0.17. A model trained without this noise can be fragile; train with quantisation or noise injection and re-measure.
- **Correlated samples**: relaxation time `τ0 ≈ 100 ns` [WP p20]; draws closer than `τ0` are correlated, so use the ESS of section 2, not the raw count, in the gradient's variance budget. `cookbook_8`/`cookbook_10` in the torx skill emulate bias quantisation and AR(1) samples.
- **Not modelled here**: the actual chip noise spectrum, per-edge coupling errors, Z1 bias range.
