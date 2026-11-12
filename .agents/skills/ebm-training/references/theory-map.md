# Theory map: Song & Kingma 2021 (arXiv:2101.03288v2)

Evidence tags: **[P §n/Eq n]** read in the supplied PDF in this session; **[own]** our run (see `results.md`); **[recall]** from memory, verify before citing.

## 1. Definitions

- `p_theta(x) = exp(-E_theta(x)) / Z_theta`, `Z_theta = ∫ exp(-E_theta(x)) dx` [P Eq 1-2]. `Z` depends on `theta`, not on `x`.
- Score: `∇_x log p_theta(x) = -∇_x E_theta(x)`; `Z` drops out [P Eq 5]. This is why Langevin and score matching never need `Z`.
- Maximising likelihood = minimising `KL(p_data || p_theta)` up to a constant [P §3].

## 2. Maximum likelihood with MCMC [P §3]

- `∇_theta log p(x) = -∇_theta E(x) - ∇_theta log Z` [Eq 3]; `∇_theta log Z = E_{x~p_theta}[-∇_theta E(x)]` [Eq 4 derivation], so the gradient of the data log-likelihood is
  `E_data[-∇E] + E_model[∇E]` : *push energy down on data, up on model samples*.
- Langevin [Eq 6]: `x_{k+1} = x_k + (eps^2/2) ∇_x log p + eps z_k`. Exact as `eps -> 0, K -> ∞`; Metropolis adjustment gives MALA.
- **CD** (Hinton 2002): init at data, few steps. **PCD** (Tieleman 2008): persistent chain. **Replay buffer** (Du & Mordatch 2019). Truncation biases the gradient [P §3, citing Schulz 2010, Fischer & Igel 2010, Nijkamp 2019].
- Bias removal: coupled-MCMC unbiased estimators (Jacob 2017; Qiu 2019); Du et al. 2020 differentiate through the sampler and add an entropy term [P §3].
- [own, 3 seeds] 8+6 RBM, exact KL after 600 steps: CD-1 0.049 +- 0.035, CD-10 0.052 +- 0.010, 1-sweep PCD 0.64 +- 0.36, 100-sweep chains 0.025 +- 0.006; CD-1's mean gradient is biased (relative excess 0.56) but CD-10's is nearly unbiased (excess 0.05) (`results.md` R4).

## 3. Score matching [P §4]

- Fisher divergence [Eq 7]: `½ E_data ||∇_x log p_data - ∇_x log p_theta||²`.
- Implicit form [Eq 8], using integration by parts (Hyvärinen 2005):
  `E_data[ ½ Σ_i (∂_i E)² ∓ Σ_i ∂_i² E ]`.
  **Errata [own, verified]:** for an energy (`p ∝ e^{-E}`) the Hessian term enters with a **minus** sign: `½||∇E||² - tr ∇²E`. The PDF prints `+`; that is the form for `log p̃`, not `E`. Test: 1D `N(0, s²)`, `E = a x²/2`: loss `½a² s² ∓ a` has its minimum at `a = 1/s²` only with `-` (with `+` the minimiser runs to the lower bound). Same flip applies to Eq 15 and 17, and to the unnumbered Taylor form in §5.1 (after Eq 26): expanding `softplus(a)+softplus(b)` with `a = E(x)-E(x-v)`, `b = E(x)-E(x+v)` gives `2 log 2 + ½[½(g·v)² - vᵀHv] + o(‖v‖²)` (g = ∇E, H = ∇²E) [own derivation: use `softplus(t) = log 2 + t/2 + t²/8 + O(t⁴)` with `a + b = -vᵀHv` and `a² + b² = 2 (g·v)²` up to higher order], whereas the PDF prints `¼ E[½(…)² + Σ ∂²E v_i v_j]`: the sign differs and so does the harmless overall prefactor. `scripts/score_matching_sign_check.py` tests the sign on the real `loss_ssm`.
- Cost: full trace of the Hessian is `O(d²)`; SM needs positive, continuous, regular density.
- **DSM** [Eq 9-10, Vincent 2011]: perturb `x~ = x + eps`; minimise `½ E ||∇ log q(x~|x) - ∇ log p_theta(x~)||²`. Gaussian noise gives target `-z/sigma`. Consistent for `q`, not for `p_data` unless sigma -> 0, where variance grows like `1/sigma²` [Eq 11-12]; control variate `c_theta` [Eq 13-14, Wang 2020].
- **SSM** [Eq 15-17, Song 2019]: project on random `v`; Hessian-vector product replaces the Hessian: cost `O(d)` [Eq 16]. With `p(v) = N(0,I)` the first term simplifies to `½||∇E||²` [Eq 17] (Hutchinson trace estimate).
- **Link to CD** [§4.3, Hyvärinen 2007]: one Langevin step CD = rescaled score matching, `(eps²/2) ∇_theta D_F + o(eps²)`; de Bruijn: `d/dt KL = -½ D_F` [P §4.3].
- **Blindness to mode weights** [§4.4]: for disjoint supports `S0, S1`, `∇ log p_data` does not contain the mixture weight `π`. Fix: multi-scale noise (NCSN, Song & Ermon 2019/2020; SDE, Song et al. 2021). [own, 3 seeds] mass in the small mode: SSM 0.126 ± 0.005, DSM 0.184 ± 0.033, NCE 0.194 ± 0.003 vs true 0.200 (`results.md`). Note the model there has Gaussian components, so the density is positive everywhere and SM is consistent in theory; the modes are 8.9σ apart, so the density at the midpoint is about e^-10 of the peak (4.4σ from each mode): a minibatch of a few hundred points essentially never samples the valley that carries the weight information. Treat the result as finite-sample near-blindness, not as the strict disjoint-support case.

## 4. Noise-contrastive estimation [P §5]

- Posterior `p(y=0|x) = p_n(x) / (p_n(x) + ν p(x))` [Eq 18-19]; maximise conditional log-likelihood [Eq 21]; optimum gives `p_theta* = p_data` [Eq 22-24] **including** `Z` as a learnable scalar.
- Noise must be near but not equal to data (Gutmann & Hirayama 2012); learned noise: ACE, CNCE, FCE [P §5].
- Special case with `p_n(x) = p_data(x - v)` reduces to SSM as `||v|| -> 0` [Eq 26-29; P §5.1].
- Practical: self-normalising when `Z` is absorbed (Mnih & Teh 2012) [P §5].

## 5. Other objectives [P §6]

- Differences/derivatives of KL (minimum velocity, minimum probability flow, minimum KL contraction): partition function cancels.
- Stein discrepancy [Eq 30]: `sup_f E_data[∇ log p_theta^T f + tr ∇f]`; kernelised (KSD) or learned `f` with Hutchinson trace.
- Adversarial / variational [Eq 31-32]: for any `q_phi`,
  `E_data[log p_theta] ≤ E_data[-E_theta] - E_q[-E_theta] - H(q_phi)` (Jensen), then `max_theta min_phi E_q[E_theta] - E_data[E_theta] - H(q_phi)`.
  `q_phi` must be cheap to sample **and** have a tractable entropy -> invertible flows, noisy neural samplers (Grathwohl 2020a), or variational bounds on `H` (Kumar 2019, Dai 2019). [own] A Torx `DiscretePCircuit` gives exact `q` and exact `H` for small n (`scripts/torx_variational_sampler.py`).
- f-divergence EBMs (Yu et al. 2020): KL, reverse KL, TV, Hellinger.

## 6. Not covered by the paper (and what to read) [P §7]

Latent-variable EBMs, downstream applications, and the recent shift to score/diffusion sampling and compositional energies: see `sources.md` (Boltzmann machines, DBM, JEM, score SDE, DTM).

## 7. Quick equivalences and relations

| Objective | What matches | Needs `Z`? | Needs samples from model? | Sees mode weights? |
|---|---|---|---|---|
| MLE (+MCMC/CD) | log p | no (gradient) | yes | only if chains mix |
| SM / SSM | `∇_x log p` | no | no | no (disjoint modes) |
| DSM (sigma) | `∇ log q_sigma` | no | no | partly (smoothing links modes) |
| NCE | `p/p_n` | learned scalar | no (noise samples) | yes |
| Stein / KSD | Stein operator | no | no | no |
| Adversarial | KL via `q_phi` | no | via `q_phi` | via `q_phi` |
