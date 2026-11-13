# Frozen contextual-vector holdout manifest

Frozen on 13 November 2026 before opening the contextual-vector test split.
This is the matched ordinary-computing control for the M2 Gaussian contextual
reference evaluated under [M2_CONTEXTUAL_HOLDOUT_MANIFEST.md](M2_CONTEXTUAL_HOLDOUT_MANIFEST.md).

## Frozen setup

- Source revision: `92dc0b7aed60e5273de8bd7b65fd3e8fb813c159`.
- Same synthetic data, train/validation/test split, fixed lexical schedule,
  five grammar-filtered negatives, Adam budget and seeds `17`, `29`, `43` as
  the Gaussian lexical stage.
- Tied SGNS: one input/output table, D=6 (`26*6 = 156` parameters).
- Untied SGNS: input plus output tables, D=3 (`26*2*3 = 156` parameters).
- Selection: validation lexical macro AUC descending, loss ascending, then
  earlier step. All six selected checkpoints are step 400.

## Frozen contextual readout

For every window, calculate `h = u_bank + mean(u_context1, u_context2)` using
the input table `u` only. Form F and R centroids from training anchors only,
compare by cosine similarity, and award half credit only to exact ties. There
is no classifier and no family label at inference.

Evaluate exactly these checkpoints:

| Variant | Seeds | Checkpoint pattern |
|---|---|---|
| tied D6 | 17, 29, 43 | `experiments/runs/contextual-vector-tied-d6-seed{seed}.npz` |
| untied D3 | 17, 29, 43 | `experiments/runs/contextual-vector-untied-d3-seed{seed}.npz` |

Report macro accuracy, 6-by-6 per-family correct counts, cross-family errors
and ties separately per seed. Do not tune dimension, learning rate, selection
rule, centroids or readout after seeing this holdout. The outcome tests whether
the toy contextual result requires an energy landscape; it does not establish
natural-text performance, sampling dynamics or thermodynamic hardware use.
