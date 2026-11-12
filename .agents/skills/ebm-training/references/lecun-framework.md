# The LeCun framework for energy-based learning

Evidence status: the 2006 tutorial (LeCun, Chopra, Hadsell, Ranzato, Huang) was **downloaded and read in this session** (sections 2.1-2.2 and 4.2, Eq. 6-25 and 45-47; tags **[L06 §n/Eq n]**). The 2022 position paper could not be retrieved (OpenReview returned HTML), so everything about JEPA/VICReg is **[recall]**: verify before citing. Song & Kingma facts are tagged [P] as in `theory-map.md`.

## 1. Vocabulary

- Energy `E(W, Y, X)`: scalar, lower = more compatible. No normalisation required. [L06 §2]
- Inference: `Y* = argmin_Y E(W, Y, X)` (or sampling from a Gibbs distribution at inverse temperature `β`, Eq. 25). [L06]
- **Energy function vs loss functional** [L06 §2.1]: inference minimises the energy over `Y`; learning minimises a *loss functional* over the energy function (over `W`). A good loss makes the energy of the correct answer lower than every incorrect answer's.
- Latent variable `Z`: `E(W,Y,Z,X)`. Reduce by **minimisation** `F_∞(Y,X) = min_Z E` or **marginalisation** `F_β(Y,X) = -(1/β) log ∫ exp(-β E) dZ` [L06 Eq. 45-47; the two coincide as `β → ∞`]. An RBM's visible free energy is `F_1` over the hiddens (closed form, `training-recipes.md`).

## 2. Loss functionals (per-sample forms) [L06 §2.2, checked against the PDF]

With `Y^i` the correct answer, `Ȳ^i` the most offending incorrect answer (lowest energy among the incorrect answers; for continuous `Y`, answers within ε of `Y^i` count as correct) [L06 §2.2], `m` a margin. Equation numbers in the tutorial: energy loss Eq. 6, perceptron Eq. 7, hinge Eq. 11, log Eq. 12, square-square Eq. 16, NLL Eq. 18-19 and 23, NLL gradient Eq. 24, Gibbs distribution Eq. 25. In the tutorial Eq. 19 `F_β = (1/β) log ∫ exp(-β E)` carries a **plus** sign, whereas its latent-variable free energy in Eq. 47 is `-(1/β) log ∫ exp(-β E)`: the two signs are inconsistent as printed, and Eq. 19 is really the log-partition term, so `L_nll = E(Y^i) + F_β`:

| Loss | Form | Pushes up on | Collapse risk |
|---|---|---|---|
| Energy loss | `E(Y^i)` | nothing | collapses unless the architecture forbids a flat energy (e.g. `E = ||Y - G(X)||²`) |
| Perceptron | `E(Y^i) - min_Y E(Y)` | the current minimiser | flat energy gives zero loss: collapse (needs margin) |
| Hinge / margin | `max(0, m + E(Y^i) - E(Ȳ^i))` | `Ȳ^i` | safe if `m>0` and contrast is informative |
| Log loss | `log(1 + exp(E(Y^i) - E(Ȳ^i)))` | `Ȳ^i`, soft margin | safe |
| Square-square | `E(Y^i)² + (max(0, m - E(Ȳ^i)))²` | `Ȳ^i` to the margin | safe, bounded |
| NLL | `E(Y^i) + (1/β) log ∫ exp(-β E(y)) dy` | **all** `y`, weighted by Gibbs probability | none (the partition term is the push-up) but needs `Z` |

The tutorial lists three further margin-type losses not tabulated here: LVQ2 (non-convex), MCE (non-convex, sigmoid-saturated) and square-exponential (Eq. 13, 15, 17; the last has an infinite margin) [L06 §2.2.3].

NLL is exactly the Song & Kingma MLE: the gradient of the loss is `∇_W E(Y^i) - E_{y~p}[∇_W E(y)]` [L06 Eq. 24]: the second term is **minus** the model expectation, so descending on the loss raises the energy of model-probable answers (equivalently, ascending the log-likelihood gives `-∇E(Y^i) + E_p[∇E]`, Song & Kingma Eq. 3-4). The tutorial also notes that NLL reduces to the perceptron loss as `β → ∞` and to the log loss for two answers [L06 §2.2.4], and that Monte Carlo and variational approximations can be read as strategies for choosing which `y` get pulled up. CD, PCD, replay buffers, and every MCMC trainer are Monte Carlo approximations of the NLL push-up; NCE and margin losses replace the push-up set by something cheaper.

## 3. Collapse and the two ways to prevent it [recall]

Collapse = the model assigns (almost) the same low energy everywhere, so it explains nothing. Two families:

1. **Contrastive**: explicitly raise energy at selected points (model samples, noise, corrupted data, hard negatives). Cost grows with the dimension of the space: the number of contrast points needed to shape a surface in high dimension grows very fast. LeCun's argument (2022) is that this is why contrastive methods scale poorly to high-dimensional continuous data.
2. **Regularised / architectural**: limit the *volume of low-energy space* without enumerating negatives: bottleneck or sparse latents, weight decay, noisy or low-dimensional latent `Z`, and for joint embeddings the VICReg terms (variance hinge keeps each embedding dimension's std above a floor; covariance penalty decorrelates dimensions; invariance term is the prediction loss).

The Song & Kingma methods sit in the first family (MLE/CD/NCE/adversarial) or in a third, *local* one (SM/DSM/SSM), which constrains the **gradient field** of `E` instead of its level. A local method cannot see global level differences between disconnected regions, which is the mode-weight blindness of [P §4.4] ([own, 3 seeds] SSM recovers 0.126 +- 0.005 of the true 0.200 mass, NCE 0.194 +- 0.003; `results.md` R3).

## 4. JEPA in this vocabulary [recall]

Joint-Embedding Predictive Architecture: encode `x` and `y` to `s_x`, `s_y`; a predictor with latent `z` outputs `ŝ_y = Pred(s_x, z)`; the energy is a distance `D(s_y, ŝ_y)` in **representation space**, not pixel space. Properties worth designing around:

- Non-generative: no density, no `Z`, no sampler. Predicting in representation space lets the model discard unpredictable detail.
- The latent `z` must be restricted (low-dimensional, sparse, noised or regularised) or the predictor can explain anything: that restriction *is* the collapse control for `z`.
- Encoders need an anti-collapse mechanism of family 2 (VICReg-style regularisers) or an asymmetric trick (stop-gradient with a moving-average target encoder, as in I-JEPA/V-JEPA - **[recall]** verify).
- Hierarchy (H-JEPA) stacks the idea at several time scales.

Use JEPA-style training when the goal is representations/prediction, not a normalised density. Use MLE/NCE when you need likelihood, a calibrated `Z`, or a Boltzmann sampler (hardware).

## 5. Boltzmann machines through this lens

- Energy `E(s) = -(Σ b_i s_i + Σ J_ij s_i s_j)` (Ising/Boltzmann machine; spins ±1); visible free energy after summing hiddens.
- NLL loss; positive phase = push-down on data (clamped), negative phase = push-up on model samples (free-running). The gradient `-(<s_i s_j>_+ - <s_i s_j>_-)` is the finite-dimensional form of the two-term identity (`scripts/gradient_identity_check.py`).
- CD-k with small k is an *approximate* push-up set (neighbours of the data): good for local shaping, blind to far spurious minima and to weights between modes.
- RBMs are tractable (bipartite) so chromatic block Gibbs is exact: that is why THRML uses two colour blocks.

## 6. Design-review questions (use before spending compute)

1. What is the energy of a correct vs an incorrect configuration, concretely, and on what scale?
2. Which loss functional, and what exactly does it push up?
3. What prevents collapse (regulariser, negatives, architecture)? What would you measure to see it failing?
4. Is the inference procedure (argmin, MCMC, k steps) the same one used at training time? If not, what bias?
5. What is the **smallest** instance on which you can compute the exact answer, and has the code passed there?
6. What claim does the final figure support, and which alternative explanation (sampler length, init, regularisation, seeds) is not yet ruled out?

## 7. Collapse detectors (cheap, model-agnostic)

- Energy spread: `std_x E(x)` over data and over negatives; `E_data - E_neg` should stabilise, not diverge.
- Representation std per dimension and effective rank of the embedding covariance (JEPA): both falling = collapse.
- Mode accounting: mass per mode against a known truth or a long run.
- Gradient check against the exact model (when small enough).
