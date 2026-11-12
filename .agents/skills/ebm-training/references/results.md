# Measured results (own runs, 2026-11-08, JAX 0.11.1 CPU, Python 3.13)

Every number below was printed by the script named in its heading. All "± sd" use the population standard deviation (ddof = 0); multiply by 1.22 for the 3-seed sample sd. Run from outside `master/`: `cd /tmp && python -I <abs path to script>`. Single machine, so wall-clock is indicative. Results are tied to the stated seed and hyper-parameters; do not read a ranking from fewer than 3 seeds unless stated.

## R1. `scripts/gradient_identity_check.py` (under a minute)

5-spin RBM (3 visible, 2 latent), exact enumeration.

| Check | Result |
|---|---|
| autodiff `∇ log Z` vs `E_p[-∇E]` | max abs diff 0 (float32). Near-tautological: it guards the energy/sign convention of your enumeration code, not the sampler |
| THRML `estimate_kl_grad` (mean of 5 keys) vs autodiff of exact KL | weights max abs error 0.0052 (4 standard errors = 0.0106), biases 0.0022 (0.0130); gradient scale 0.582 |
| descend / ascend along the THRML gradient | exact KL 0.4488 -> 0.2575 (descend) vs 1.1182 (ascend): sign convention confirmed |
| closed-form spin-RBM free energy vs enumeration | max abs diff 6.4e-7 |

## R2. `scripts/score_matching_sign_check.py` (about 10 s)

True precision 2.000 (Gaussian, s² = 0.5). Closed-form algebra: `-d2E` recovers 1.995, `+d2E` (as printed in the tutorial) runs to the grid edge, 0.050. The **real** `loss_ssm` of the zoo (jvp Hessian-vector product, 2D, 20000 points): `SSM_SIGN=-1` recovers 2.000, `SSM_SIGN=+1` gives 0.050.

## R3. `scripts/continuous_ebm_zoo.py --seeds 3 --steps 3000` (about 5 min per method, 3 seeds, mean ± sd)

Target: two Gaussians (σ 0.45) at (-2,0) and (2,0), weights 0.8 / 0.2, modes 8.9σ apart (midpoint density about e^-10 of the peak). Model: MLP 2-128-128-1, softplus. Exact KL on a 220x220 grid over [-5,5]². Defaults: energy L2 `reg` 0, replay restart 0, SSM sign -1, NCE noise σ 2.5, Langevin 30 steps, eps 0.3.

| Method (flags) | KL(true‖model) | mass in small mode (true 0.200) |
|---|---|---|
| `nce` (σ_noise 2.5) | 0.0156 ± 0.0009 | 0.194 ± 0.003 |
| `cd_k` (reg 0) | 0.0153 ± 0.0019 | 0.186 ± 0.014 |
| `mle_cd` (persistent buffer, restart 0) | 0.0091 ± 0.0020 | 0.218 ± 0.018 |
| `ssm` (sign -1) | 0.0226 ± 0.0035 | 0.126 ± 0.005 |
| `dsm` (σ 0.3) | 0.0649 ± 0.0039 | 0.184 ± 0.033 |

Sweeps (same protocol, 3 seeds):

| Change | KL | mass small mode |
|---|---|---|
| `cd_k` reg 0 / 0.01 / 0.2 | 0.015 / 0.153 / 0.680 | 0.186 / 0.459 / 0.483 |
| `mle_cd` restart 0 / 0.05 / 0.2 | 0.009 / 0.986 ± 0.046 / 2.171 ± 0.054 | 0.218 / 0.001 / 0.000 |
| `nce` σ_noise 0.8 / 1.5 / 2.5 / 4.0 | 4.139 ± 0.564 / 0.0132 ± 0.0007 / 0.0156 / 0.0213 ± 0.0023 | 0.120 / 0.199 / 0.194 / 0.187 |
| `ssm` sign +1 (as printed) | 43.96 ± 1.06 | 0.820 |

Readings (3 seeds, one model, one toy problem):
1. Everything except SSM recovers the mixture weight to within about ±0.03 under the defaults; SSM is consistently low (0.126 ± 0.005), consistent with the near-blindness of score matching to weights across a low-density valley [P §4.4]. DSM with a single σ is noisier (0.184 ± 0.033, individual seeds 0.149-0.228): not distinguishable from 0.200 at 3 seeds.
2. The energy L2 regulariser, usually recommended, **hurt** here: it pulled both modes towards equal weight (0.46-0.48) and raised KL ten- to fifty-fold. With a correct Langevin noise term and gradient clipping, reg 0 was stable. Earlier in development a CD run without the regulariser diverged, but that run also used a decoupled (tiny-noise) sampler, so the cause was not isolated: do not cite it.
3. Restarting 5% or 20% of replay-buffer chains from uniform noise **collapsed** the buffer onto the large mode (mass 0.001 / 0.000) while no restart worked best. Chain-length sweep (`--lang-steps`) at restart 0.05: 30 steps KL 0.986 ± 0.046, mass 0.001; 100 steps KL 0.508 ± 0.053, mass 0.007 ± 0.002; 300 steps (1 seed) KL 0.183, mass 0.042. At restart 0 with only 5 steps: KL 0.383 ± 0.173, mass 0.599 ± 0.107. So the outcome depends strongly on the Langevin steps per update (more steps help the restarted buffer but 300 still did not recover the weight; 5 steps ruin even the no-restart buffer). This is consistent with, but does not prove, the hypothesis that noise-restarted chains are too short to reach the modes; a 4096-point buffer in 2-D also accumulates thousands of Langevin steps per slot, which is a confound for any comparison with high-d image EBMs. The common advice to restart ~5% (Du & Mordatch) is therefore setup-dependent: verify per-mode mass.
4. NCE needs the noise broad enough to cover the data: σ_noise 0.8 (about the data scale) failed (KL 4.1) with a mis-learned `log Z`.
5. Wrong SSM sign: KL 44 on every seed.
6. NCE learned `log Z` vs exact (last seed of each noise setting): sigma_noise 0.8 learned 0.01 vs exact 5.50 (wrong, KL 4.1); 1.5 learned -0.03 vs -0.02; 2.5 learned -0.02 vs 0.01; 4.0 learned -0.02 vs 0.01.

## R4. `scripts/thrml_rbm_negative_phase.py` (3 seeds, 600 Adam steps lr 0.02, batch 128; about 6 min)

RBM 8 visible + 6 latent, exact enumeration (16384 states), data = 4 prototypes with 8% bit flips (entropy 3.446 nats). CD chains start at `(v = data, h ~ p(h|v))` (see the ordering note in `thrml-torx-mapping.md`). Final exact `KL(p_data‖p_model(v))`, mean ± sd over 3 seeds:

| Estimator | final KL | per seed |
|---|---|---|
| CD-1 | 0.049 ± 0.035 | 0.016, 0.097, 0.035 |
| CD-10 | 0.052 ± 0.010 | 0.057, 0.038, 0.062 |
| PCD (1 sweep, 128 persistent chains) | 0.640 ± 0.358 | 0.302, 0.484, 1.136 |
| long (100 sweeps, fresh chains) | 0.025 ± 0.006 | 0.034, 0.020, 0.023 |

Gradient bias at a mid-training point (exact KL 0.290), 24 independent estimates, **all estimators with 2048 negative chains and 2048 data points** (equal statistical power): relative L2 error of the mean vs the autodiff gradient of the enumerated KL, Monte Carlo floor `‖std/√R‖/‖g‖`, and excess `√(bias² − floor²)`:

| Estimator | rel. bias | MC floor | excess | cosine |
|---|---|---|---|---|
| CD-1 | 0.571 | 0.087 | 0.564 | 0.824 |
| CD-10 | 0.115 | 0.104 | 0.048 | 0.995 |
| PCD (chains burned in 300 sweeps at this fixed θ, then 1 sweep per estimate) | 0.174 | 0.105 | 0.139 | 0.987 |
| long (100 sweeps) | 0.078 | 0.101 | 0.000 | 0.997 |

Readings: (a) CD-1's gradient is clearly biased (excess 0.56, cosine 0.82) yet its training still reached KL 0.05 here, so a biased gradient does not imply a bad fixed point on an easy 14-spin problem; (b) CD-10 is nearly unbiased (excess 0.05); (c) the long-chain estimator is consistent with unbiased down to the noise floor of about 0.10; (d) PCD at a *fixed* θ has only a small excess (0.14; its floor is optimistic because successive estimates share chains), yet PCD was the worst *trainer* (0.64 ± 0.36). So the failure is not the instantaneous gradient bias: the likely cause is that 128 persistent chains lag the moving parameters at lr 0.02 (hypothesis; not tested with a smaller lr or more chains). A previous version of this script initialised the CD chains from biases only (not from the data) and reported "CD-1 diverges, KL 2.1 -> 18": that was a bug, not a property of CD. An earlier bias table also gave PCD/long only 128 chains against CD's 2048 (floors 0.29 / 0.32 vs 0.10), which made PCD look much worse (0.72); that comparison was unfair and is replaced above.

## R5. `scripts/torx_variational_sampler.py`
n = 5 spins, Torx `StateVectorSimulator.density` gives exact `q_phi` and exact entropy. `indep` = 5 PNOT gates; `pairs` = 5 PNOT + 10 PCNOT.

| Seed | KL(q‖p) indep | KL(q‖p) pairs | student KL exact MLE | student KL q_indep | student KL q_pairs |
|---|---|---|---|---|---|
| 0 | 1.368 | 0.303 | 0.0025 | 0.422 | 0.880 |
| 1 | 0.424 | 0.383 | 0.0000 | 0.445 | 0.203 |
| 2 | 0.468 | 0.073 | 0.0003 | 1.263 | 0.645 |

Readings: the pairwise circuit always fits a fixed Boltzmann target better (inner step), but the end-to-end student is not consistently better (3 seeds, wide spread) and ends 0.2-1.3 nats of KL above the exact-negative baseline (which reaches at most 0.0025 nats), so the residual `KL(q‖p)` bias dominates. With the first settings (inner 10, lr_theta 0.05) the student diverged (KL 7 and 21): the adversarial loop needs `q` to keep up with `theta`. This script uses exact expectations under `q` (no sampling noise), so it measures the best case for the approach.

## R6. `scripts/mcmc_mixing_diagnostics.py --rounds 300 --matched --key-seed {0,1,2}` (about 4 min each, 32 chains)

8+6 RBM with coupling scale 7 (several deep basins), exact visible law by enumeration; statistic = visible magnetisation. ESS is the **multi-chain** estimator (pooled variance including between-chain variance, Geyer pairs). Three independent chain seeds on the same model, values listed per seed:

| Sampler | split R-hat | ESS | TV(visible) | iid TV floor |
|---|---|---|---|---|
| plain block Gibbs, β = 1, 300 rounds | 6.28 / 5.12 / 7.22 | 16.8 / 17.2 / 16.4 (of 9600) | 0.086 / 0.179 / 0.043 | 0.007 |
| plain block Gibbs, 1200 rounds (compute-matched to PT) | 3.37 / 3.62 / 3.55 | 18.5 / 19.4 / 18.6 (of 38400) | 0.147 / 0.073 / 0.122 | 0.004 |
| parallel tempering, β ∈ {1, 0.55, 0.3, 0.15}, 300 rounds (4x sweeps) | 1.035 / 1.039 / 1.029 | 1049 / 829 / 1038 (of 9600) | 0.020 / 0.008 / 0.026 | 0.007 |

PT neighbour swap acceptance [0.89-0.90, 0.52, 0.36-0.37] (top pair above the 0.2-0.8 rule: that rung could be spaced wider). Readings: quadrupling the plain sampler's compute did not fix mixing (R-hat 3.4-3.6, ESS barely moved), so PT's gain is not just more sweeps. TV of the trapped sampler is **not monotone in run length** and varies 4x across chain seeds (0.043-0.179): it counts which basins the 32 chains landed in, so R-hat and multi-chain ESS carry the conclusion, not TV. PT reached the iid TV floor on one seed (0.008 vs 0.007) and was within 0.02 of it on the others. Correction history: a first version used a within-chain ESS (each chain centred on its own mean) and reported IAT 2.0 / ESS 1563 for the trapped sampler at 100 rounds; that estimator hides trapping.

## R7. `scripts/hardware_robustness.py` (about 1-2 min)

Trained 'long' RBM (exact KL 0.0291; parameter range R = 1.35). Assumptions [ASSUME]: one shared range R, mid-tread quantiser uniform in parameter space, bias noise Gaussian. Exact KL after quantising biases and couplings to n bits: 8 bits 0.0311, 6 bits 0.0227, 5 bits 0.0311, 4 bits 0.1196, 3 bits 0.2527 (reference spread: 20 random uniform ±step/2 perturbations of 6-bit size give exact KL 0.0325 ± 0.0055, range 0.023-0.043, so differences below about 0.01 are within what rounding-size noise alone produces; 6 bits is *better* than the baseline). After Gaussian bias noise δb (20 draws, mean ± sd): 0.02 → 0.0296 ± 0.0037; 0.05 → 0.0362 ± 0.0094; 0.10 → 0.0630 ± 0.0245; 0.20 → 0.1742 ± 0.0877. One model; not a chip prediction.

## R8. `scripts/ais_logz.py` (about 1.5 min; 256 chains per replicate, 8 replicates, linear β path from the uniform distribution)

Forward AIS (stochastic lower bound) and reverse AIS from exact samples (stochastic upper bound), mean ± sd over replicates, vs the enumerated log Z:

| Coupling scale (exact log Z) | K=10 | K=100 | K=1000 | K=5000 |
|---|---|---|---|---|
| 1.5 (15.712): forward | 15.729 ± 0.072 | 15.719 ± 0.015 | 15.714 ± 0.007 | 15.710 ± 0.002 |
| 1.5: reverse | 15.705 ± 0.072 | 15.723 ± 0.024 | 15.709 ± 0.006 | 15.713 ± 0.003 |
| 7.0 (60.788): forward | 60.670 ± 0.710 | 60.772 ± 0.081 | 60.794 ± 0.019 | 60.788 ± 0.010 |
| 7.0: reverse | 61.896 ± 0.477 | 60.811 ± 0.092 | 60.779 ± 0.021 | 60.787 ± 0.008 |

The exact value lies within the bracket at 3 standard errors in every cell. At low K the bracket is wide for the strongly coupled model (K=10: 60.67-61.90 around 60.79; sd of the forward estimate 0.71, 0.081, 0.019, 0.010 for K = 10, 100, 1000, 5000). Rugged cases (`--scales 25,60 --ks 100,1000,5000 --reps 4`, 256 chains): scale 25 (exact 213.522) forward 213.472 / 213.507 / 213.525, reverse 213.539 / 213.558 / 213.524; scale 60 (exact 510.507) forward 510.294 / 510.602 / 510.462, reverse 510.807 / 510.469 / 510.495 for K = 100 / 1000 / 5000. **What failure looks like** (`--scales 25 --ks 10,100,1000 --reps 8 --chains 4`): K=10 forward 196.349 ± 7.584 and reverse 220.174 ± 0.966 (a 24-nat bracket around 213.522), K=100 212.994 ± 0.774 / 214.366 ± 1.515, K=1000 213.412 ± 0.311 / 213.692 ± 0.353: few chains and few temperatures give a lower bound 17 nats low with an 8-nat replicate sd. The bracket-contains-truth test cannot fail by being too wide, so always read the width and the replicate spread, not just the check. Caveats: one 14-spin model per scale; the reverse chain starts from *exact* samples, which is only possible because the model is enumerable (use RAISE for real models); all our 14-spin RBMs, even at coupling scale 60, anneal easily with 256 chains; a regime where the bracket stays wide at large K was not found at this size, so the failure regime of large or frustrated models remains undemonstrated here. Single-run orderings (forward above exact at K=10 for scale 1.5) are within the replicate spread.

## Reproducing

```bash
cd /tmp
python -I /c/Users/MuhamadDanielHakimiB/Desktop/master/.agents/skills/ebm-training/scripts/gradient_identity_check.py
python -I .../scripts/score_matching_sign_check.py
python -I .../scripts/continuous_ebm_zoo.py --seeds 3 --steps 3000   # sweeps: --reg --restart --ssm-sign --nce-sigma
python -I .../scripts/thrml_rbm_negative_phase.py
python -I .../scripts/mcmc_mixing_diagnostics.py --rounds 300
python -I .../scripts/hardware_robustness.py
python -I .../scripts/ais_logz.py
python -I .../scripts/mcmc_mixing_diagnostics.py --rounds 300 --matched --key-seed 1   # key-seed 0,1,2 in R6
python -I .../scripts/torx_variational_sampler.py --seed 0
```
`--quick` on the zoo and RBM scripts is a smoke test (not for numbers).
