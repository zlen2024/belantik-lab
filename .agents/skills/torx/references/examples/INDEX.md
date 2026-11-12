# Examples: which script proves what

Run from OUTSIDE `master/` (`cd /tmp && python <path>/<script>`). `check_examples.py` runs all of them and asserts the quoted numbers.
Runtime = wall time on the development machine (JAX CPU). Quick = included in `check_examples.py --quick`.

| Script | Proves / shows | Used by | Runtime | Quick |
|---|---|---|---|---|
| `cookbook_1_core.py` | `jit(grad)` over `build_circuit` + `expval_all`; 3 `diff_method`s vs exact; `sample`/`sample_multiple`; pdit softmax; MSB index order | SKILL.md, gradients.md, gates.md | 10 s | yes |
| `cookbook_2_hybrid.py` | `HybridPCircuit` regime switching; `AffineGaussianSimulator` propagate/condition vs analytic | factors-dfg-hybrid.md, Example D | 10 s | yes |
| `cookbook_3_factors.py` | custom factor, DFG, `param_key` sharing, `TiledFactor`, `ChainFactor` | factors-dfg-hybrid.md, Example C | 5 s | yes |
| `cookbook_4_attention.py` | p-bit dot-product variance formula; non-saturated softmax TV vs N (t-CI); DAC bias; TV floor; score-error correlation | estimator-math.md | 2 min | no |
| `cookbook_5_gibbs.py` | one-site Gibbs kernel (`PReset` + `PNOT`); chromatic Gibbs TV vs multinomial floor | explain-code.md Example B | 10 s | yes |
| `cookbook_6_gradients.py` | unbiasedness at several theta (K=2, K=3); variance vs N; `_filter` bias; silent zero gradient; REINFORCE | gradients.md | 20 s | no |
| `cookbook_7_api.py` | Gaussian autodiff vs finite differences; `log_var=-inf` routes; `draw_circuit`; `to_thetas`; `DFGInfo`; custom gate; `log_probability`; pathwise gradients; JumpDiffusion | gates.md, explain-code.md F | 20 s | yes |
| `cookbook_8_attention_thrml.py` | THRML attention with n-bit DAC; one-hot softmax prototype; correlated-sample N_eff | estimator-math.md | 1 min | no |
| `cookbook_9_variance.py` | variance formulas for `_inf`, `_single`, `_filter` (single gate); equal-budget theta sweep; depth; REINFORCE baseline bias; K=3 `_filter` bias | gradients.md | 1 min | no |
| `cookbook_10_key_sampling.py` | end-to-end key-index sampling (THRML), 12 seeds, DAC range | estimator-math.md | 3 min | no |
| `cookbook_11_train.py` | training loop; Jensen bias vs S and the `2 Var(f)/S` law; baselines; discrete score term; truth tables; `filter_jit` timing; vmap | gradients.md, gates.md | 2 min | no |
| `cookbook_12_study.py` | equal-cost training; interior-target biased fixed point; failure cases with real messages; `condition` arithmetic; nested DFG; Pdit tables; Torx vs THRML with Welch test | gradients.md, explain-code.md | 2 min | no |
| `cookbook_13_bias_theory.py` | per-gate law `[L'' + (1/2) L''' g_j] Var(f)/S` for `_single`, `(1/2) L''' g_j Var(f)/S` for `_inf`, delta-method `_filter` bias and variance (1e); unified K-branch / reps form (1f); exact binomial-sum `_filter` bias with empty branches for a quadratic loss, incl. an `m0 != 0` circuit and an empty branch 0 (1g); independent keys (bit, sum, difference readouts, cubic and exp losses); small-S validity (1b); k-batch remedy (1d); degenerate chain limit; fixed points `t`, `t - 1/S`, `t/(1 - 1/S)`; paper Eq. 44 on a Boltzmann conditional gate, with a downstream gate and at equal cost incl. the `d/da`-only parameter-shift row (3c) | gradients.md, Example I | 1 min | no |
| `torx_learn.py` | PNOT as a p-bit; gate tables; PISING pair; sampled dot product vs N | gates.md | 30 s | no |
| `z1_attention.py` (THRML) | independent-spin dot product, methods A and B; attention TV | estimator-math.md | 1 min | no |
| `onchip_gate.py` (THRML) | 2-local AND gate fails | wiki 09 | 2 min | no |
| `check_examples.py` | regression runner: 107 value checks, source-bound doc-number check, mutation catch rate | SKILL.md | 9 min (`--quick` 30 s) | |
| `doclock.py` + `doc_numbers.lock.json` | pins every doc number to one printed position or source literal; `check_examples.py --relock` rebuilds it (review the diff), normal runs only read it | SKILL.md rule 4 | part of the doc check | |
