# Mapping EBM training onto THRML and Torx

Run Python from outside `master/` (the `thrml/` and `torx/` folders shadow the packages). Read the `thrml` and `torx` skills first; this file only adds the EBM-training layer. Everything marked [own] was run (`results.md`).

## 1. Concept -> object

| EBM concept | THRML | Torx |
|---|---|---|
| Energy `E(s) = -β(Σ b s + Σ J s s)` | `IsingEBM(nodes, edges, biases, weights, beta)` | `PISING` (state-vector only) or a directed circuit |
| Positive phase (data clamped) | `IsingTrainingSpec` positive program, `data_blocks` clamped | n/a |
| Negative phase (model samples) | negative program, free blocks, `init_state_negative` | `circuit.sample` from a directed `q_phi` |
| MLE gradient | `estimate_kl_grad` -> `(grad_w, grad_b, pos_moments, neg_moments)` | `StateVectorSimulator.expval_all` + `jax.grad` (exact) |
| CD-k / PCD / long chains | choose `init_state_negative` and `schedule_negative` | single-site Gibbs: `PReset` + `PNOT(2βl)` |
| Sampler `q_phi` with tractable `H(q)` | n/a | `StateVectorSimulator.density` (exact) |
| Block Gibbs colouring | two blocks for RBM / bipartite | n/a |

## 2. CD, PCD and long chains in THRML (verified patterns) [own]

All three use the same `estimate_kl_grad`; only the negative initialisation and schedule change (`thrml_rbm_negative_phase.py`):

```python
sched_neg = SamplingSchedule(n_warmup=k, n_samples=1, steps_per_sample=1)    # k sweeps, one sample
spec = IsingTrainingSpec(model, [Block(vis)], [], [Block(lat)], [Block(vis), Block(lat)], sched_pos, sched_neg)
# CD-k : init_neg = [data, h0] with h0 ~ p(h | data)    # NOT hinton_init for lat: see the ordering trap below
# PCD  : init_neg = persistent state carried from the previous step
# long : init_neg = hinton_init(...) with k ~ 100
```

- **Ordering trap (cost us a wrong result).** THRML updates free blocks in list order. With `free_blocks = [Block(vis), Block(lat)]` the first half-sweep resamples `vis` from the initial `lat`, so `init=[data, hinton_init(lat)]` throws the data away and the CD-k chain is a k-sweep run from bias-initialised noise. Initialise `h0 ~ p(h|data)` (`sample_hidden_given_visible` in the script) or list the latent block first. The thrml skill does not document this.
- **Persistent chain end-state without reimplementing the sampler.** `estimate_kl_grad` returns moments but not states. Reproduce its key derivation (`_, key_neg = split(k_grad, 2); keys = split(key_neg, n_chains)`) and call `sample_states` with the same per-chain keys: the trajectory is identical (`sample_states` and the moment accumulator both go through `sample_with_observation`). The script asserts the match (max discrepancy 0.0). This depends on THRML internals: re-check after upgrading THRML.
- `init_pos` must be an array per positive-sampling block with shape `(n_chains_pos, batch, block_len)`; `[]` is the recurring bug.
- Rebuild `IsingEBM` and the spec every step (they are immutable and parameter-dependent); under plain `jax.jit` pass arrays and close over structure (see the `thrml` skill).
- `optax.update` then `apply_updates` already descends KL; do not subtract again [own: ascend gave KL 1.118 vs descend 0.258].

## 3. Sampler audit with THRML

To test whether the training chain is long enough: take the trained `(b, J)`, run `sample_states` for thousands of sweeps from many starts, compare empirical `<s_i>`, `<s_i s_j>` and the visible histogram against (i) exact enumeration when n is small, (ii) the short-run statistics used in training. A large gap means the model is a short-run generator.

## 4. Torx as the amortised sampler (Eq. 31-32)

`scripts/torx_variational_sampler.py` [own]: `q_phi` = `DiscretePCircuit([PNOT(i)...] + [PCNOT([i,j])...])`, `q = sv.density(sv.build_circuit(circ, thetas), x0)`, entropy `-Σ q log q` exact, `jax.grad` through it exact. Notes:

- Site 0 is the most significant bit; Torx `P(1) = sigmoid(θ)`, spin mean `tanh(θ/2)`, THRML bias mean `tanh(b)`: convert (`θ = 2 atanh(m)`).
- PNOT-only `q` is a product distribution; adding PCNOT pairs gives correlations but not a general joint (an XOR-flavoured parametrisation). Expressivity is the limit of the whole method: residual KL(q‖p) 0.07-0.30 for 15 gates on 5 spins (3 teachers).
- Exact-density simulators scale as `2^n`; beyond ~20 sites use `BranchingSimulator` samples and score-function estimators (see the `torx` skill: sampled `diff_method`s are biased in different ways).
- `jax.grad` through `sample()` returns zeros without error; only `expval` / `density` paths carry gradients.
- Adversarial loop stability [own]: inner 10 / lr_theta 0.05 diverged; inner 30 / lr_theta 0.01 stable.
- For Boltzmann sampling prefer single-site conditional kernels over bond-local `PISING` walks (biased; `torx` pitfall 4).

## 5. When to use which

| Goal | Use |
|---|---|
| Train an Ising/RBM by maximum likelihood, n large | THRML `estimate_kl_grad` + long enough chains |
| Check any gradient code | exact enumeration (this skill) + THRML estimate |
| Train a model with a **tractable** normalised likelihood to sample for you | Torx directed circuit (exact `log q`) |
| Evaluate gradient estimators for a stochastic circuit | `torx` skill `gradients.md` |
| Large cloud run | `extropic` skill (`extro-sim`); keep one active job at a time (memory note) |
| Hardware realism (degree ≤ 16) | `thrml` graph + `check_degree` as in `attn_thrml/ebm_attn_baby` |

## 6. Not yet exercised here (be honest in reports)

- THRML with real image data (MNIST sized models) was not run in this skill; the RBM here is 14 spins.
- Torx score-function (REINFORCE) training of `q_phi` at n > 20 was not run.
- Hardware (Z1/XTR-0) behaviour is only probed by `hardware_robustness.py` (parameter quantisation and bias noise on one trained RBM); chip noise spectrum, per-edge errors and Z1 ranges are not modelled.
