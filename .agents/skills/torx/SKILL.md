---
name: torx
description: Write, explain and verify Torx code (Extropic's open-source JAX library, package `extro-torx`, `from torx import psc`) for parametrised stochastic circuits (PSC) and directed factor graphs (DFG). Use when the user mentions Torx, PSC, DFG, pbit/pdit/pmode gates (PNOT, PCNOT, PSWAP, PJUMP, POR, PISING, AffineGaussianGate, MixtureGaussianGate), StateVectorSimulator / BranchingSimulator / AffineGaussianSimulator, parameter-shift or score-function gradients through a sampler, p-bit estimators (dot product, attention), or Extropic X0/XTR-0/Z1 programming above the THRML layer. Also use to explain existing Torx code step by step. For raw Ising / block-Gibbs / Boltzmann-machine training use the `thrml` skill.
---

# Torx

Torx is to THRML what PyTorch is to CUDA: nodes of a DAG are stochastic transition kernels with trainable parameters. Stack: Torx -> Thermalizers (compiler, unreleased) -> THRML -> backend (XTR-0 now, Z1 in 2027, GPU simulators). Whitepaper arXiv:2608.01612; local repo `master/torx/` (own git, treat as upstream); notebooks `master/torx/examples/`.

## Environment (read first)

- Installed editable (`pip install -e master/torx`). Python 3.11+. `optax` is an optional extra.
- **Never run Python with `master/` as cwd**: the `torx/` repo folder shadows the package (`ImportError: cannot import name 'psc'`). Use another cwd and absolute script paths.
- Top-level imports only: `from torx import psc`; factors and DFG: `from torx import AbstractReferenceFactor, DFG, Site, TiledFactor, ChainFactor`.
- Git Bash `/tmp` and Windows Python temp differ; give Python absolute Windows or Git-Bash-translated paths consistently.

## Core model (verified)

- Primitives: pbit {0,1}, pdit {0..d-1}, pmode R^N. A gate is a column-stochastic kernel `K(y|x)`; a circuit is the product of layers (paper Eq. 16).
- Two-branch gate: `G(theta) = (1 - sigmoid(theta)) I + sigmoid(theta) B` (Eq. 17), `theta` shape `(1,)`, stored as a **logit**. Gates with K>2 branches use `softmax([0, theta])`, `theta` shape `(K-1,)`.
- Parameters: **list, one entry per gate** (`len(thetas) == len(gates)`); `circuit.init_params(key)`.
- p-bit as a spin: start 0, `PNOT(theta)` gives `P(1) = sigmoid(theta)`, spin mean `m = tanh(theta/2)`, so `theta = 2*atanh(m)` (Torx logit convention; THRML biases use `m = tanh(theta)`).
- State-vector index order: site 0 is the most significant bit.
- Simulators: `StateVectorSimulator` (exact, differentiable, `2^n`), `BranchingSimulator` (sampled, `num_samples`, `diff_method`), `AffineGaussianSimulator` (exact Gaussian moments; input is a point mass). The paper's `SampleSimulator` / `HybridSampleSimulator` do **not** exist in this repo.

## Snippets (run these before trusting variants)

```python
import jax, jax.numpy as jnp
from torx import psc
circ = psc.DiscretePCircuit([psc.PNOT(0), psc.PCNOT([0, 1]), psc.PNOT(1)])
th = [jnp.array([0.3]), jnp.array([-0.4]), jnp.array([0.8])]
sv = psc.StateVectorSimulator()
loss = lambda t: sv.expval_all(sv.build_circuit(circ, t), jnp.zeros(4).at[0].set(1.0))[1]
jax.grad(loss)(th)                                      # exact: -0.0373, -0.0524, 0.1153

bsim = psc.BranchingSimulator(num_samples=4000, diff_method="param_shift_inf")
sloss = lambda t, k: bsim.expval_all(bsim.build_circuit(circ, t), jnp.zeros(2, jnp.int32), k)[1]
jax.jit(jax.grad(sloss))(th, jax.random.key(0))         # sampled, same structure
```

`examples/cookbook_1_core.py` runs this and shows the `circuit.sample`, `sample_multiple`, pdit-softmax and index-order checks.

## Gradients (details in `references/gradients.md`)

- Only `expval`/`expval_all` are differentiable. `sample()` is not. A loss on joint statistics (`E[s_i s_j]`, cut value, attention score) needs the exact density (small `n`) or a score-function estimator you write; Torx has no REINFORCE code.
- `jax.grad` through `sample()` returns **exact zeros with no error**; always compare a new gradient with the `StateVectorSimulator` gradient on a tiny circuit.
- Three `diff_method`s with different cost, variance, bias and `reps` limits; the full table, formulas, the theta-dependent equal-cost ranking and the choice rule are in `references/gradients.md`. One-liners: `_inf` unbiased, `reps==1`; `_single` Eq. 40 variance, K=2, `reps==1`; `_filter` ratio estimator, the only one with `reps>1`, biased at tiny N with rare branches and, for nonlinear losses, at small S with a gate-dependent sign (delta-method formula in `gradients.md`; its sign follows `v1 - v0 - p Delta^2` for a quadratic loss: opposite to `_single` for a single gate from a fixed input and on the 3-gate chain and the 6-gate last gate, the same sign on the 2-gate circuit); paper Eq. 44 energy-based gradient is verified in plain JAX (`cookbook_13`).
- `param_map` is **not an API** (notebook 08 numpy index array). Tying: `Site.param_key`, `TiledFactor`, `ChainFactor`, `reps`.
- `build_circuit` inside `jit` is fine (that is how `torx/tests/test_gradients.py` works); a compiled circuit passed into `jax.jit` fails, use `equinox.filter_jit`.
- A loss nonlinear in a sampled expectation (`(E_hat - t)^2`) has a biased gradient for `_single` (bias `[L'' + L'''(E - E'_j)/2] Var(f)/S`: `2 Var(f)/S` on every gate for a quadratic loss, gate-dependent otherwise) and for `_filter` a gate-dependent bias `[L'' p(1-p)(v1 - v0 - p Delta^2) + (1/2) L''' g_j Var f]/S` (for a quadratic loss its sign follows `v1 - v0 - p Delta^2`: opposite to `_single` for a single gate from a fixed input, `E* = t/(1 - 1/S)`, and on the 3-gate chain and the 6-gate last gate; the same sign on the 2-gate circuit; valid only when every branch is sampled; with rare branches use the exact binomial sum for a quadratic loss); for a quadratic loss `_inf` is unbiased and the two-batch product removes the `_single`/`_filter` bias; for other nonlinear losses all three are biased at O(1/S) (`_inf` least: `(1/2) L'''(E) dE/dtheta_j Var(f)/S`), a degree-k polynomial needs k independent batches, and non-polynomial losses need large S (`gradients.md`).

## Pitfalls (each is sourced or reproduced)

1. `PISING` with `J != 0` changes marginals (J>0 aligns, J<0 anti-aligns); it cannot compute a product (`cookbook`/`torx_learn.py`: `<ab> = +0.055` vs target -0.24 at J=0.3). It is state-vector only.
2. There is no AND/XOR/multiply gate. `POR` clears its second bit. `PDEMUX` branch 0 is not identity.
3. Monte Carlo error is `1/sqrt(N)`; a TV against an exact law has a finite-sample floor that **depends on the target distribution** (256 states, N=6000: 0.082 for the uniform worst case in `cookbook_4_attention.py`, 0.047 for a Boltzmann ring in `cookbook_5_gibbs.py`). Compute it by simulating `multinomial(N, P)` from your own target (`cookbook_5_gibbs.py`) before claiming bias.
4. Bond-local `PISING` walks are biased for Boltzmann sampling; use single-site conditionals (`PReset` then `PNOT(2*beta*l)`), as in `cookbook_5_gibbs.py`.
5. `log_var = -inf` (deterministic Gaussian channel): gradients are finite through `AffineGaussianSimulator`, but `d/dlog_var` is **NaN through `HybridPCircuit.sample`** (A and b stay finite; `-30` gives 1.5e-7). Clamp to about -30 if you differentiate `log_var`. `condition` on a deterministic observed site may need `jitter>0` (`cookbook_7_api.py`).
5b. Through `HybridPCircuit.sample` continuous parameters get pathwise gradients (`Displace`, `Scale`, `A`, `b`, mixture `means`/`log_vars` of the selected component); the discrete branch choice (`PditCycle` theta) gets zero gradient, so the total gradient is **biased** unless you add the score-function term `x * d log p(branch)/d theta` (`cookbook_11_train.py`: analytic `[-0.1224, -0.4177]`, pathwise `[0, 0]`, with score term `[-0.1223, -0.4171]`).
6. Attention test must not saturate the softmax (weights like `[2e-6, ..., 0.999995]` test nothing).
7. nb09 REINFORCE uses the exact `2^n` density; it does not scale.

## Explain-code protocol (summary; full version in `references/explain-code.md`)

Sites -> gates and branch operation -> kernel (`p = sigmoid(theta)`) -> equation-to-code mapping -> simulator and why -> readout, estimator and standard error -> gradient path and bias -> departures from the ideal equation. Always show a hand-computed number next to the printed number (worked example: three-gate circuit with gradients `0.1153, -0.0524, -0.0373`, `cookbook_1_core.py`).

## When to read what

| Need | File |
|---|---|
| Gate semantics, theta shapes, constructors, index order | `references/gates.md` |
| Estimators, `diff_method` table, REINFORCE, training loop, tying | `references/gradients.md` |
| p-bit dot-product variance, softmax error vs N, DAC bias, TV floor, key-norm mapping | `references/estimator-math.md` |
| Walking through a program, two worked examples | `references/explain-code.md` |
| Custom factors, DFG, `param_key`, Tiled/Chain, hybrid, Gaussian simulator | `references/factors-dfg-hybrid.md` |
| Z1 and X0 facts with source tags | `references/hardware-facts.md` |
| Torx vs THRML decision | `references/thrml-vs-torx.md` |
| Background notes and paper walk-through | `references/wiki/index.md`, `references/wiki/paper/index.md` |
| Which script proves what, runtime, quick flag | `references/examples/INDEX.md` |

## Maintaining the wiki (part of this skill)

Canonical sources: the English files `references/*.md` (checked by `check_examples.py`) are authoritative for every number and formula. The wiki in `references/wiki/` (Bahasa Melayu, English technical terms) is the learning notes and the paper walk-through: it summarises and links to the references, and when it quotes a number it must cite the script. Do not keep another copy anywhere else.

1. After reading source, running code or finding an error, update the relevant page **and** add one row to that folder's `log.md` (date, action, file, note).
2. Every folder under `wiki/` must have `index.md` and `log.md`. Adding, renaming or deleting a page means updating `index.md`.
3. Tag each fact: source (whitepaper page, notebook, blog), own run (script name), or assumption. Update the "tahap bacaan" and "status" columns in `wiki/index.md` when depth changes.
4. New numeric claims need a script in `references/examples/` that reproduces them; run it from outside `master/`. `check_examples.py` (about 13 minutes; 107 value checks) also checks the numbers in the prose, tables and bullets of all English reference files and SKILL.md (code blocks and headings excluded and counted) in three layers:
   - **Lock (the strict guard, `doclock.py`).** Every decimal, every integer with 3 or more digits (or comma grouped) and every `at step N` is pinned in `doc_numbers.lock.json` to ONE position: a printed line of the script its paragraph cites (identified by the line's text with numbers masked, its occurrence and the number's index) or a numeric literal in that script's source (an input). Each run reads the value at exactly that position and compares it at the doc's precision, so a wrong number fails even when the same value is printed elsewhere (a target or input value written in place of a measured result), and so does a script whose output drifts. Realistic substitution test (each doc number replaced by other values printed or set by its script): all caught by the lock, about 70% by the context check alone (the runner prints both rates). Workflow after editing a doc number or a script: `--relock` (with `--docs-only=DIR` or a fresh run) prints every added, rebound, removed and REFUSED entry with its old and new position and context score. Relock is guarded: if the locked position still prints the old value, a changed doc number is REFUSED and the old entry kept (the script did not change, so the edit contradicts the evidence); a rebinding to a weaker position (a source literal in place of printed output, or a lower context score) and a new number with no shared context are refused too. Override one entry only with `--force-relock=FILE:TOKEN` after checking it. Old and new docs are matched by aligning each file's sequence of numbered lines (difflib), not by line text plus occurrence, so relabelling or rewording one row of a repeated table keeps every other row's entry; a table row whose first cell is a plain integer (S, steps, N) must have that label printed on its locked line, so swapping row labels or adding a row that was never run is refused (an equal value bound to an unlabelled line is rebound to the labelled one and listed). A built-in lock self-test runs on every check against a frozen snapshot of the fixed-point table (`LOCK_FIXTURE`), not the live docs, so editing the docs never breaks it: four cosmetic edits (relabel one row, another row, all rows of the repeated block, insert a paragraph) must give 0 refused and 0 removed; five attacks (wrong value in a row, reworded row with a wrong value, fabricated row, swapped row labels, reworded prose with a wrong value) must be refused and keep failing. If cookbook_13 stops printing the snapshot's numbers the self-test is skipped with a WARN (update the snapshot). A locked number deleted from the docs fails the normal run until relocked, and relock lists it as removed. Never force or relock to silence a failure you have not explained.
   - **Context check (relock-time sanity).** A lone number must be printed on a line that shares context with the doc (a nearby word, the table row label, or a neighbouring doc number), and a source literal counts only where the doc writes an input (`theta=`, `S=`, `lr`, `below`, `[`). Keyed results (`p =`, `t =`, `at step`) must be printed with the same key. Comma/slash groups must appear in order on one printed line. Numbers inside inline code other than its final `= result` are shown arithmetic and are checked leniently.
   - **Source check for `[WP pN]` facts.** Every number in a cell or sentence tagged `[WP pN]` (1-2 digit integers included) must occur on the cited pages of the local whitepaper PDF (pypdf), together with its unit when the doc gives one (`5%`, `8-bit`, `degree-16`, a time with its `ns`); a bare number must not be satisfied only by a citation marker such as `[9]` or by the page number itself. If pypdf or the PDF is missing the runner FAILS (set `TORX_WP_PDF` when the skill is copied elsewhere, or pass `--no-wp` to skip knowingly).

   Exemptions are listed with a reason in `ALLOW_REASONS` (paper and notebook values, arXiv ids, derived SE multiples). Known limits: the lock pins a binding chosen by context score, and some values appear on several equally matching printed lines, so for those the lock guards the doc value but may watch a different (equal) printed copy (`--relock --verbose` lists them); `[BLOG]` facts and 1-2 digit integers outside `[WP]` cells and `at step` are not machine-checked; small-perturbation mutation rates are also printed (the runner fails below 85% at 30 units). `--quick` runs five fast scripts in about 30 s; `--save=DIR` stores outputs and `--docs-only=DIR` re-runs only the doc layers. `references/examples/INDEX.md` maps each script to what it proves.
5. Never state a notebook result as verified unless the cell was run.

## Honest status

Notebook prose 01-04 and `basic_usage` were read closely; 05-16 were read for introductions, headings and conclusions only. Notebook code was **not** executed, except where a cookbook script re-implements it (15 factors, 06 Gibbs). Z1 numbers other than degree-16/chromatic Gibbs come from the blog, not the whitepaper.
