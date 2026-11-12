# Reproducible ML experiments

## Establish the evaluation protocol

- Define the hypothesis, primary metric, baseline, data unit, and available compute budget before tuning. Preserve official benchmark splits; otherwise establish a held-out test set before data-dependent model development.
- Split by the true independent unit: patient, user, device, source document or time as appropriate. Use stratification for relevant class balance, group-aware folds for repeated entities, and chronological evaluation for forecasting. Check duplicates across splits where plausible.
- Fit normalization, imputation, feature selection, resampling and other learned preprocessing on training data only, inside each cross-validation fold. Use validation data for early stopping, thresholds and hyperparameter selection.
- Reserve test evaluation for a frozen model/protocol. During development report test as **not evaluated**; do not inspect it every run. If test feedback has already influenced choices, disclose that and qualify the estimate or obtain an untouched holdout where feasible.

## Run in bounded stages

Check a small batch and baseline first, then scale within the requested budget. Use fair baselines with comparable data, preprocessing and tuning opportunities. Change variables deliberately; retain failed and negative runs that affect interpretation. Do not expand into unrequested sweeps or cloud jobs.

Each substantive run should record:

- Run ID, hypothesis, exact command/config and seed(s).
- Dataset version and split definition or IDs/hash, preprocessing, and code revision (or a snapshot when there is no Git repository).
- Dependency versions, device, runtime and relevant resource constraints.
- Train and validation metrics on comparable evaluation protocols, generalization gap when meaningful, and held-out status. Report test metrics only for the frozen evaluation.
- Checkpoint/artifact paths, warnings/failures and the conclusion supported by that run.

Use the project's existing tracking system or a small JSON/Markdown record; a hosted service is optional. Report variability across independent runs or an appropriate uncertainty estimate when claiming an improvement. One seed does not establish robustness. Do not invent a comparable training metric when regularization/augmentation makes it incomparable; explain the difference.

## Investigate suspicious results

For unexpectedly strong scores or unstable folds, check leakage, duplicates, feature availability at inference, label alignment, split logic, class balance and metric computation. Check predictions manually on a small sample. Use shuffled-label or other negative-control checks only when informative. For anomaly detection, keep threshold tuning separate from test labels and include metrics suitable for imbalance, not accuracy alone.

For generative/EBM tasks, distinguish task quality, likelihood estimates or proxies, energy values, and sampler mixing. Low energy or low reconstruction loss alone does not establish sample quality or a valid probability model. Report relevant chain diagnostics and estimation limitations.

Conclude with measured comparisons, uncertainty and limitations. If a run cannot finish, preserve the partial evidence and state exactly what remains unmeasured.
