# Repeated SWORDS development probe: active-window R

21 November 2026. This is a posthoc evaluation of the previously inspected SWORDS development inputs, labels and fixed checkpoints. It uses the same 219 eligible target/candidate sets and 132 informative targets as `SWORDS_EXTERNAL_CANDIDATE_PROBE_RESULTS.md`. No model training, Gaussian baseline, new THRML simulation, SWORDS test or physical hardware was used.

In each candidate-specific star, target, all available context token occurrences and the candidate are active. Score is raw `R=(1/4)Σ E[s_target,a s_candidate,a]` under the exact joint law. Candidates are scored in different systems; adding the candidate also changes the per-spoke scaling. The uniform-context control retains the same number of spokes and their strength while replacing each contextual unary with a uniform one. Zero coupling retains unary fields, so its raw spin correlation need not be zero. All new scores were checked finite and within [-1,1]; the two controls were verified against independent pair-law identities for each scored target, candidate and seed.

| Readout | Mean target-macro pairwise AUC, 3 seeds |
|---|---:|
| Active-window R | 0.58132 |
| R with uniform context | 0.57367 |
| R with zero coupling | 0.56379 |
| Matched vector target + context, previous run | 0.62272 |
| External-candidate Ising affinity, previous run | 0.59274 |
| Frequency | 0.47151 |

Active R minus vector is -0.04140, source-cluster paired bootstrap 95% interval [-0.08442,+0.01047], seed differences -0.02948, -0.04339, -0.05133. Active R minus uniform-context R is +0.00765 [-0.01756,+0.02445], with seed differences -0.00877, +0.01976, +0.01196. Both intervals include zero and these are conditional descriptive intervals after repeated development use.

Excluding the pre-identified dominant source leaves 55 targets. Active R is 0.54197 versus vector 0.58660, difference -0.04463 [-0.09713,+0.03559]. Active R minus uniform-context control is -0.00091 [-0.02232,+0.03276]. No useful contextual gain or advantage over the vector is demonstrated on this subset. An interval crossing zero is not evidence of equivalence.

The earlier SWORDS audit found that all 87 eligible targets excluded for no contrast had no acceptable candidate remaining in the small vocabulary. Therefore this AUC also does not represent performance across the full candidate-generation task. Sources are clustered and one dominates. These observations apply to this model/checkpoint/readout and do not invalidate all possible energy landscape approaches.

The protocol and code were frozen as `bed8701` before calculation. MLflow experiment `word-energy-natural-context-evaluation`, run `3878d013f4ae477f8af4c1e41cf1252b`, records source artifact hashes, predictions, per-target AUC, controls and bootstrap. Prior THRML reanalysis checked only an older checkpoint and two three-token illustrative stars; it does not verify these full-softmax checkpoints or all context lengths. Independent critic review follows separately.
