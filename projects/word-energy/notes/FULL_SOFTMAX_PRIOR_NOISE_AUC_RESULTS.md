# Historical noise AUC comparison — 21 November 2026

The full-softmax log-count checkpoints were scored on the same WikiText validation windows and 20 sampled noise candidates used in `WIKITEXT2_FOUR_SPIN_RESULTS.md`. The metric is pairwise noise AUC averaged per article and then across articles. All three historical margin checkpoints and the exact earlier AUCs were reproduced. The three new checkpoint hashes and dataset hashes were verified. Evaluation is posthoc on a reused development panel.

| Seed | Historical margin, original offset | New full-softmax, training offset | Historical margin, full log-count offset |
|---|---:|---:|---:|
| 17 | 0.70067 | 0.68992 | 0.69238 |
| 29 | 0.69666 | 0.68952 | 0.69075 |
| 43 | 0.70631 | 0.69178 | 0.69805 |
| Mean | 0.70121 | 0.69041 | 0.69373 |

New minus historical original: -0.01080, conditional paired article bootstrap 95% [-0.01695, -0.00541], seed deltas -0.01075, -0.00714, -0.01452. With full log-count offset for both checkpoints, new minus old is -0.00332 [-0.00548, -0.00119]. The primary comparison changes training objective and offset. The common-offset comparison is a posthoc readout diagnostic. Bootstrap resamples 60 documents after averaging seed differences and does not quantify uncertainty across new datasets or training populations.

For the lowest-frequency third of validation centers, window-weighted noise AUC averaged over seeds falls from approximately 0.512 to 0.391. The highest-frequency third rises from approximately 0.876 to 0.941. These descriptive strata use the frozen train counts, and sampled noise can include valid words or ties. The result suggests stronger frequency bias in this noise discrimination task, not established causal attribution or a general semantic finding.

The earlier full-vocabulary document-macro MRR on the same WikiText validation improved from 0.19448 for corrected margin to 0.19800 for new full-softmax. That metric ranks against 2,000 vocabulary entries, unlike this 20-noise pairwise AUC. The two findings can coexist. No reserved test or new training was performed.

MLflow experiment `word-energy-full-softmax-prior-ablation`, run `9e0a8ff84d364bf3b3ab852ce24cc2e5`, artifact `auc_comparison.json`. Script `scripts/evaluate_prior_ablation_noise_auc.py`.
