# Four-spin WikiText-2 experiment: results

Date: 2026-11-19. Development experiment; no reserved test set used.

## Question and design

Does a four-spin landscape improve prediction and contextual representations over the earlier two-spin model? This comparison changes parameter count, connectivity and maximum cross-interaction scale together; it cannot establish an effect caused by spin count alone. Each word has four binary spins, 16 configurations, four fields and six internal couplings. A three-word window has 12 spins. Joint scores use exact factorization of the 4,096-state model. 16 states are not 16 learned meanings.

Same dataset run `f1b7db5231ed4a5788b566e388f8e731`, train-only vocabulary 2,000, 200,000 training examples, 10,000 validation examples across 60 articles, same seeds 17/29/43, sampling, 3 epochs, and objective. Results are development measurements, not independent confirmation. Prior checkpoints reused. Frozen training commit `12d05ab`. Summary MLflow run `c05a1d631d814a9db373876c83d7d413`; WiC development run `1e43fb7e38524085b4807478b0dd503f`. Details, raw scores, hashes, diagnostics and checkpoint references are in their config artifacts.

Joint = shared interacting context; pair_add = independently scored pair model; vector = shared vector baseline; shuffled = context shuffled control. AUC higher is better for distinguishing observed pairs from sampled noise; MRR measures ranking among sampled candidates. Macro AUC averages article scores.

## All runs, last epoch

| Model | Seed | Parameters | Macro AUC | MRR | MLflow run ID |
|---|---:|---:|---:|---:|---|
| joint4 | 17 | 20004 | 0.70067 | 0.33330 | `457aaa1ab2af46678844386986a61b32` |
| joint4 | 29 | 20004 | 0.69666 | 0.32954 | `1cca3e70769b42a7b8b5e64cf5987b08` |
| joint4 | 43 | 20004 | 0.70631 | 0.33038 | `b1de614b15144b9e98bb2a5e70584b31` |
| joint4_shuffled | 17 | 20004 | 0.64900 | 0.28582 | `069bd8f936c941129aee311d0af5aaa6` |
| joint4_shuffled | 29 | 20004 | 0.64890 | 0.28665 | `0a7ad8261ef84c0cadc4cab349795ba2` |
| joint4_shuffled | 43 | 20004 | 0.64787 | 0.28611 | `287d769711ba4d489b24d23b2f117da8` |
| joint | 17 | 6002 | 0.69337 | 0.31835 | `d4ac69067db94b9ba6e9bee33e6d35da` |
| joint | 29 | 6002 | 0.69545 | 0.32127 | `5459599bea1042d7ba4cc6cc9dd5e8e7` |
| joint | 43 | 6002 | 0.69164 | 0.31731 | `1ca78c0254ef45688c6ab4dc850c99d9` |
| joint_shuffled | 17 | 6002 | 0.64728 | 0.28595 | `dccc22087adb4af19e1c3d5a84a4bbaa` |
| joint_shuffled | 29 | 6002 | 0.64768 | 0.28644 | `c9288b36529a48b2bb39729fe6a25fc2` |
| joint_shuffled | 43 | 6002 | 0.64815 | 0.28624 | `06ec285b31e44e3a8aae818fd6378788` |
| pair_add4 | 17 | 20004 | 0.70552 | 0.33206 | `e939ae16e2aa4fa0a0baaf7fb846e8c8` |
| pair_add4 | 29 | 20004 | 0.70098 | 0.33136 | `655a436a788f49abb4b3500b35ab930a` |
| pair_add4 | 43 | 20004 | 0.70528 | 0.32872 | `0dc398804071468d92247d7d8504bd8d` |
| pair_add | 17 | 6002 | 0.69197 | 0.31642 | `cb6a51e8a6c045fe8d8c9d7600a800f0` |
| pair_add | 29 | 6002 | 0.69385 | 0.32080 | `df570c9a41524b67865f761176181693` |
| pair_add | 43 | 6002 | 0.69191 | 0.31473 | `92eba2d279114bb7b9da63a4c543a052` |
| vector10 | 17 | 20000 | 0.72717 | 0.35067 | `bf13c1856c884633adbea51758024039` |
| vector10 | 29 | 20000 | 0.72428 | 0.35105 | `07ab333a95e6489e8a0c37b5d88832a3` |
| vector10 | 43 | 20000 | 0.72836 | 0.35346 | `d6981991a7bc4871b01e21246bdd79b3` |
| vector | 17 | 6000 | 0.70141 | 0.32668 | `238d4bdaa10b4d949c4cf7c5b162bc12` |
| vector | 29 | 6000 | 0.70333 | 0.32570 | `d4b280542ad84888983e65f6d96c2d46` |
| vector | 43 | 6000 | 0.69873 | 0.32669 | `073d6ac07df149a8ade0be8ec257e527` |

## Paired macro AUC differences

Intervals resample validation articles conditional on these trained models; they do not represent uncertainty over all datasets or independent model training.

| Comparison | Difference | Conditional 95% interval |
|---|---:|---|
| joint4_minus_joint | +0.00773 | [+0.00312, +0.01282] |
| joint4_minus_joint4_shuffled | +0.05262 | [+0.04287, +0.06275] |
| joint4_minus_pair_add4 | -0.00272 | [-0.00535, +0.00013] |
| joint4_minus_vector10 | -0.02539 | [-0.03081, -0.02027] |
| pair_add4_minus_pair_add | +0.01135 | [+0.00662, +0.01582] |
| vector10_minus_vector | +0.02544 | [+0.01962, +0.03153] |

Joint4 is better than the earlier joint2 on this reused panel, but has more parameters and a different graph and coupling scale. Joint4 does not establish superiority to its matched pairwise alternative; the 10-dimensional vector with nearly matched parameter count scores higher. Shuffled control scores much lower, consistent with usefulness of the real context for this task.

## Landscape and sense checks

The independent critic recomputed all four-spin diagnostics: across seeds, 74.97% of words already had multiple strict one-spin-flip unary minima at initialization, 86.83% after joint training, and 97.17% in the shuffled control. This diagnostic is not a count of meanings and does not establish learned polysemy. Joint4 average posterior entropy decreased 2.5243 to 0.9653; mean posterior-to-unary total variation rose 0.1045 to 0.3048. The additive model has similar behavior, so this does not establish a unique joint-context advantage.

WiC is a repeated development probe, with 167 of 638 examples retained using the prior fixed rule; it is not a fresh reserved test. It covers only a subset, hence cannot represent all words. Similarities use the context-conditioned target-state distributions; the additive checkpoint is evaluated with a joint-star counterfactual readout, rather than its native additive score.

| Probe | AUC |
|---|---:|
| joint4:epoch3 | 0.52994 |
| joint4:initial | 0.59149 |
| joint4_shuffled:epoch3 | 0.53711 |
| pair_add4:epoch3 | 0.52879 |
| two_spin:joint:epoch3 | 0.54362 |
| two_spin:joint:initial | 0.54892 |
| two_spin:pair_add:epoch3 | 0.57470 |
| two_spin:previous:bag_jaccard | 0.56377 |
| two_spin:vector:epoch3 | 0.55519 |
| vector10:epoch3 | 0.58508 |
| vector10:epoch3:context_only | 0.55505 |
| vector10:initial | 0.59632 |
| vector10:initial:context_only | 0.54366 |

Joint4 versus initial difference: -0.06155 (95% target-cluster bootstrap interval [-0.13449, +0.01569]); versus joint2: -0.01368 (interval [-0.06934, +0.04554]). The intervals are wide and cover zero. Training did not demonstrate improved word-sense discrimination on this limited reused probe.

## Interpretation and next action

The four-spin predictive model performs better than the previous two-spin setup on this panel, but the vector baseline benefits more and joint interactions do not beat the matched additive model on macro AUC. Do not scale up on the assumption that extra wells automatically encode meaning. Investigate the predictive objective and sampling reliability; then design a separately specified fresh evaluation and fair capacity/coupling ablations before any broad semantic or hardware claim. All experiments here run on a conventional CPU.
