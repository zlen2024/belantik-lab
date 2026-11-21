# Repeated WiC development probe for full-softmax checkpoints

## Scope

The 4-spin Ising and matched 10D vector full-softmax checkpoints were evaluated without WiC training using the same target-posterior and vector-cosine readouts, labels and 167 selected examples previously used for the margin checkpoints. The original validation has 638 examples, with 82 same-sense and 85 different-sense among the selected 167. This is repeated development, not an untouched benchmark. The selection excludes many OOV/invalid-context cases and covers 151 target lemmas. Contexts have one to four in-vocabulary words whereas model training used two; readouts were not trained for WiC. There is no physical-hardware measurement.

## Results

| Readout / checkpoint | AUC, mean of 3 seeds | Per seed 17 / 29 / 43 | 95% target-cluster bootstrap |
|---|---:|---|---|
| Ising full-softmax | 0.56882 | 0.55925 / 0.60072 / 0.54648 | [0.49919, 0.64233] |
| Ising previous margin | 0.52994 | 0.50502 / 0.48895 / 0.59584 | [0.45821, 0.60301] |
| Vector full-softmax | 0.57561 | 0.58350 / 0.56141 / 0.58192 | [0.48859, 0.66570] |
| Vector previous margin | 0.58508 | 0.58537 / 0.59383 / 0.57604 | [0.49400, 0.67460] |
| Jaccard word-overlap control | 0.56377 | One deterministic control | [0.48454, 0.64419] |

Paired Ising full-softmax minus Ising margin is +0.03888, 95% target-cluster interval [-0.01561, +0.09012]. Compared with matched full-softmax vector it is -0.00679 [-0.06396, +0.04748]. Compared with word overlap it is +0.00505 [-0.06792, +0.08016]. These intervals include zero. They reflect resampling target words with these trained models and this selected development subset, not uncertainty over all corpora or retraining outcomes. The result does not establish a sense improvement or an Ising advantage; the scores are also compatible with chance-level discrimination.

## Audit and provenance

The script verifies the source reference and hashes of dataset and checkpoints, checks the historical helper snapshots and reproduces the historical Ising and vector AUCs to 1e-14. It preserves the old selected inputs, labels, context ordering, cosine tie handling and bootstrap implementation. No new examples were excluded. Each run has 163 distinct predicted scores across 167 examples. The scores, source-run IDs and limitations are saved in MLflow experiment `word-energy-full-softmax-matched-vector`, run `e63956d576a8487ebcffd2f3d1521385`, artifact `wic_probe.json`. Protocol and script were committed as `1fa941b` before this repeated scoring.
