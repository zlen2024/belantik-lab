# SWORDS external-candidate development probe

21 November 2026. Protocol and code were frozen at commits `a99c00c` and `6956bca` before judgments were processed. Frozen WikiText full-softmax checkpoints were compared without SWORDS training. The primary Ising readout is the Bhattacharyya affinity of the target's exact context-conditioned spin posterior and an **external** candidate's unary state distribution. It is neither the original same-active-system correlation R nor a measured chip output. The tied 10D vector control uses target-plus-mean-context cosine. Other controls use the same eligible candidates. Gaussian has no matched natural-text checkpoint here.

## Data and metric

Of 370 SWORDS v1.1 development targets, 219 satisfy the fixed input, source and 2,000-word vocabulary conditions and contain 3,038 candidate IDs. After removing `UNSURE` votes, 132 targets contain at least one majority-acceptable and one other candidate, and 87 have no class contrast. This label-dependent metric availability is reported explicitly; it is not a label-blind sample of the entire benchmark. There are 16 source-file proxies among the 132, with 77 targets from `Nathans_Bylichka.txt`. The metric is pairwise AUC averaged within target and then across targets; it is not an official SWORDS generation score. Scores and eligibility were saved before judgments were processed. The official test was not opened.

The independent critic found that all 87 eligible targets without a class contrast have **no majority-acceptable retained candidate**. Within the 3,038 retained candidate IDs there are 232 majority-acceptable and 2,806 non-acceptable candidates. The 132 informative targets include 232 acceptable candidates and 1,621 non-acceptable candidates, yielding 2,862 within-target positive/negative pairs. Thus the conditional AUC omits a substantial coverage failure and must not be read as full-task performance.

Across all 370 development targets, the critic counted 1,393 majority-acceptable target–substitute IDs after excluding candidates matching the target surface or supplied target lemma. This denominator includes multiword/OOV candidates and targets excluded by the input pipeline. Only 232/1,393, or 16.65%, survived the complete frozen eligibility pipeline. IDs are specific to a target and are not counts of distinct word types. This is a posthoc pipeline coverage analysis, not an official SWORDS score.

## Results

| Readout / control | Target-macro AUC, mean over 3 seeds |
|---|---:|
| Ising target + context affinity | 0.59274 |
| Ising target only | 0.58737 |
| Ising context only | 0.54397 |
| Ising external-candidate pair kernel, secondary | 0.58014 |
| Matched vector target + context cosine | 0.62272 |
| Vector target only | 0.61200 |
| Vector context only | 0.57554 |
| Vector target + context dot product, secondary | 0.62276 |
| Candidate frequency only | 0.47151 |

Ising affinity minus vector contextual is -0.02998, with paired source-cluster bootstrap 95% interval [-0.05497, +0.05949] and seed deltas -0.00530, -0.04841, -0.03624. Ising affinity minus its target-only control is +0.00537 [-0.01973, +0.01589], seed deltas +0.00514, +0.00373, +0.00723. Intervals include zero. These are conditional on fixed seeds, available targets, source clusters and the exploratory protocol, and do not establish superiority or equivalence.

Removing the pre-identified dominant source leaves 55 informative targets across 15 source proxies. On this subset Ising is 0.58599, vector 0.58660, and the paired Ising-minus-vector difference is -0.00061 [-0.06130, +0.08293]. The Ising contextual-minus-target-only difference is -0.00837 [-0.02347, +0.01710]. Source concentration materially changes the point estimate and limits precision; the leave-one-source-out result remains exploratory.

For the 132 informative targets, the contextual affinity changed the sign or tie status of approximately 10.6% of candidate-pair orderings compared with target-only, averaged over seeds and targets. A change of ordering does not demonstrate that context helps. The average candidate unary entropy was 1.203 nats and target posterior entropy 1.133 nats; affinity can depend on distribution concentration. The Ising and vector learned scorers had no positive/negative score ties in the computed comparisons. Frequency had a small tie rate. The exact CPU scoring loop took about 3.3 seconds for all three seeds and all controls after loading; it repeatedly recalculates distributions and is not an optimized or hardware latency benchmark.

## Reproducibility and limitations

MLflow experiment `word-energy-natural-context-evaluation`, run `6bfbcadfb039410dbc206f963b50d2bf`, contains source snapshots, label-blind `input_manifest.json`, label-blind predictions saved before using judgments, candidate vote counts, per-target results and `summary.json`. Source data remain local and retain the official CC-BY-3.0-US attribution described in the [SWORDS repository](https://github.com/p-lambda/swords). The exact posterior/candidate scorer passed independent 4,096-state numerical and invariance checks. Contexts contain one to four in-vocabulary words although training used two; all candidate scores are classical CPU post-processing. The reused WikiText checkpoints, vocabulary restriction, non-independent source groups, omitted Gaussian and unvalidated THRML sampling limit interpretation. Independent critic review is required before any further decision.
