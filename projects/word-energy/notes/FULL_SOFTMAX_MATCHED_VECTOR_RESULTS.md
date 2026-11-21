# Matched vector full-softmax comparison — 21 November 2026

## Question and design

Does the four-spin joint-energy model outperform a tied vector with approximately the same parameter count when training inputs, loss and schedule match? The vector has 20,000 learned values and the Ising model 20,004. Both used the same 2,000-word WikiText vocabulary, train windows, orders, full-vocabulary cross-entropy, log-count offset, batch, three epochs and seeds 17/29/43. Existing Ising checkpoints were reused. This comparison addresses one tied vector baseline, not all embedding methods or matched computational cost.

Both validation panels have already been examined repeatedly. The system uses exact classical CPU scores with a host-side frequency prior; it is not a thermodynamic chip experiment. Reserved tests remain closed. The 20-noise AUC is a distinct criterion from full-vocabulary MRR, and neither alone establishes meaningful multiple senses.

## Results

| Development panel | Ising macro MRR | Matched vector macro MRR | Frequency prior | Ising minus vector, paired 95% article bootstrap |
|---|---:|---:|---:|---|
| WikiText, 60 articles | 0.19800 | 0.19810 | 0.18264 | -0.00010 [-0.00410, +0.00374] |
| EWT, 287 documents | 0.12611 | 0.13483 | 0.10575 | -0.00872 [-0.01789, -0.00112] |

On WikiText, seed differences are -0.00034, +0.00136 and -0.00132. On EWT, they are -0.00691, -0.00817 and -0.01109. The planned exploratory advance condition fails. Intervals resample documents conditional on the three trained seeds and selected panels. On the WikiText 20-noise task, Ising macro AUC is approximately 0.6904 and matched vector 0.7079. Both produce better MRR than frequency alone on these panels.

An additional descriptive frequency-strata audit uses the same previously established center-frequency thirds and sampled noise. For rare centers, Ising AUC is approximately 0.391 versus vector around 0.43; both are approximately 0.94 for the most frequent third. These are window-weighted summaries, and sampled noise may contain valid alternatives or ties. They do not establish the cause or a population-level difference.

The paired MRR gate and matched vector result do not establish the user objective of a reliable replacement embedding. The alignment critic reviews the evidence and next action separately. The Gaussian baseline has not been matched to this natural-text task, and THRML precision and rare-event parity are not established by this experiment.

## Reproducibility

Before training, the protocol and code were committed in `ab87e41`. Each run checks model scores against the existing scorer, finiteness of gradients, dataset hashes, Ising checkpoint hashes, EWT window/group hashes, and exact rowwise reproduction of stored Ising ranks on both panels. Checkpoints and rank arrays are MLflow artifacts. `word-energy-full-softmax-matched-vector` contains seed 17 `ba3d1cfad0214f36ac30c9cf66c43b23`, seed 29 `282b6343c23b443ea852d7ef4076aafa`, seed 43 `f28f6fbb35c848d8b2947c448cb61f2d`, summary `2d349f15243544218e6b97fb91830306`, and frequency audit `fbe920f4a54646d09d8e2480b1e06273`. Scripts and protocol are preserved in source snapshots.
