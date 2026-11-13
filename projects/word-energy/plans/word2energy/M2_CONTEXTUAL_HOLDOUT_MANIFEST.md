# Frozen M2 contextual holdout manifest

Frozen on 13 November 2026 before opening the Gaussian contextual test split.
This manifest records a **classical reference evaluation** only; it makes no
sampler, THRML, Torx or physical-hardware claim.

## Frozen implementation and data

- Source revision: `55dfc9e54270c26ec5cb711bfa2f5941340e8c6e` (`Repair Gaussian sense region boundaries`).
- Model: D=1, M=2 normalised Gaussian mixtures; length-3 complete active window; RBF coupling `lambda=1`, length scale `1`.
- Seeds: `17`, `29`, `43`.
- Protocol hash: `ced585cb88b56356f37c885a970a6202ee7c1739585c9cdbe8bdaf3830c49a7c`.
- Schedule hash: `da8217cf662c87b468a14c22685ff9ced64f93cd34543364e2d43aa9b55abc87`.
- Test split: exactly the synthetic holdout in the frozen manifest. It has not been used for checkpoint selection, capacity selection or numerical tuning.

## Checkpoints and controls

Evaluate the following selected checkpoints without retraining:

| Seed | Joint checkpoint | Frozen lexical control |
|---:|---|---|
| 17 | `experiments/runs/gaussian-joint-seed17.npz` | `experiments/runs/gaussian-lexical-seed17.npz` |
| 29 | `experiments/runs/gaussian-joint-seed29.npz` | `experiments/runs/gaussian-lexical-seed29.npz` |
| 43 | `experiments/runs/gaussian-joint-seed43.npz` | `experiments/runs/gaussian-lexical-seed43.npz` |

The joint selector is frozen as validation macro sense accuracy descending,
then validation total joint loss ascending, then earlier checkpoint. The
selected joint checkpoint is step 100 in all three runs. Each selected M2
joint and lexical checkpoint passed the repaired validation numerical audit
with maximum region mass delta below `4.4e-7` (gate: `1e-4`).

## Fixed readout and report

- Build `q_F` and `q_R` from **training anchors only** for the checkpoint being evaluated.
- Define the F region strictly by `q_F(z) > q_R(z)`. Equal density receives half credit.
- Use adaptive central integration split only at genuine density crossings, Gauss--Hermite context orders 8 and 12, domains `[-12,12]` and `[-15,15]`.
- Report, separately for joint and frozen-lexical checkpoints: macro sense accuracy, per-family exact counts/confusion, correct-region posterior, lambda-0 prior, gain, and all three numerical sensitivities.
- Do not select a new checkpoint, region threshold, capacity or hyperparameter after seeing the holdout.

The holdout result answers the prespecified synthetic contextual sanity check.
It cannot by itself establish a joint-training advantage over the frozen
lexical control, natural-text performance, sampler mixing or hardware
feasibility.
