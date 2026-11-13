# Word Energy research

This is the active workspace for Word Energy Embedding. The first implementation stage now contains exact references for both designs, a THRML simulator parity fixture, an explicit MLflow run contract, the frozen synthetic corpus generator, and an exact-gradient Ising ranking precheck. Neither model has been trained on natural text or executed on physical hardware.

- [Primary Ising/Z1 plan](plans/word2energy/WORD_ENERGY_Z1_FIRST_PLAN.md)
- [Gaussian baseline plan](plans/word2energy/WORD_ENERGY_EMBEDDING_V2_PLAN.md)
- [Original ideas and conversation](plans/word2energy/RUMUSAN_IDEA_BIKAR_ENERGY_EMBEDDING.md)
- [Ising review](plans/word2energy/reviews/z1_first/REVIEW_LOG.md) and [Gaussian review](plans/word2energy/reviews/energy_embedding_v2/REVIEW_LOG.md)
- [Experiment tracking decision](TRACKING_DECISION.md): the user selected local MLflow.

Implementation entry points:

- `word_energy/energy_embedding/`: normalised Gaussian-mixture landscape baseline, analytic overlap, joint-window energy and a small grid reference.
- `word_energy/ising/`: exact discrete Ising/Z1-first reference and THRML six-spin simulator bridge.
- `scripts/run_reference_smoke.py`: classical numerical smoke artifact.
- `scripts/run_thrml_fixture.py`: THRML simulator parity artifact.
- `scripts/write_synthetic_manifest.py`: deterministic 84-window synthetic corpus split.
- `word_energy/ising/exact_training.py`: exact-enumeration F2/F3 contrastive ranking pilot. It freezes grammar-filtered negatives, two cross-pool corruptions per window, batch schedules and validation checkpoint selection. It is deliberately not sampler/CD/PCD training.
- `scripts/train_ising_exact_smoke.py --full --seed 17 --evaluate-test`: runs one frozen-budget precheck and explicitly opens its holdout test split only after validation selection. The generated checkpoint and JSON are local experiment artifacts, not source.
- `scripts/log_ising_exact_smoke.py --full --seed 17`: logs an existing precheck artifact to the explicitly configured local MLflow database.
- `word_energy/energy_embedding/train_lexical.py`: constrained D=1/M=2 Gaussian-mixture lexical baseline using differentiable, analytic overlap. It deliberately stops before joint-window inference.
- `scripts/train_gaussian_lexical.py --seed 17 --evaluate-test`: runs the matching Gaussian lexical precheck; `scripts/log_gaussian_lexical.py` records the artifact in local MLflow.
- `gaussian_hermite_joint_reference(...)`: exact-for-quadrature D=1 joint free-energy and shared-window relevance reference. It is the numerical gate for later Gaussian joint training or sampling, not a trained context model.
- `scripts/train_gaussian_joint.py --seed 17`: D=1/M=2 differentiable Gauss--Hermite joint-ranking precheck. It selects by validation joint loss and records an order-8 versus order-12 smooth free-energy/gradient refinement audit; it does not yet perform sense-region evaluation or sampling.

Current status: three exact-gradient synthetic prechecks (seeds 17, 29 and 43) have been recorded locally. Seeds 17 and 43 reached the toy sense/retrieval gate, while seed 29 did not; therefore the three-seed Ising pilot does **not** pass its preregistered gate. The matching untrained random control exposed a further weakness of the small sense metric: random seed 17 already achieved 10/12 sense accuracy while its lexical AUC was only 0.3125. Thus the readout must be compared with the controls, not interpreted from the threshold alone. The frozen lexical-positive permutation control retained the original joint supervision as specified: seed 17 remained at 1.0 sense accuracy, while seeds 29 and 43 were 0.5. This isolates neither the whole supervision nor a hardware mechanism; it only tests the added lexical-positive contribution. Sampler audit and physical-hardware mapping are still outstanding. The local MLflow experiment `word-energy-exact-training` is the record of individual runs.

The lexical Gaussian baseline reached a holdout retrieval AUC of 1.0 for each of the same three seeds. This does not compare contextual sense handling: it has no Gaussian joint system yet. Its MLflow experiment is `word-energy-gaussian-lexical`.

The Gaussian joint lane now has one seed-17 numerical precheck in `word-energy-gaussian-joint-quadrature`: its held-out test remains unopened. The trained checkpoint passed the smooth quadrature refinement check (order 8 to 12: free-energy change about `2.7e-7`, mean-gradient relative change about `2.7e-6`). This validates the fixed quadrature calculation at that checkpoint, not semantic sense separation, MCMC mixing, thermodynamic hardware execution, or a multi-seed result.

The first validation-only train-anchor region evaluation reported macro sense accuracy `1.0` and context gain `0.089`, but its discontinuous region mass changed by `0.00114` from quadrature order 8 to 10. It therefore fails the planned `1e-4` region-refinement gate and remains **numerically inconclusive**; the held-out Gaussian test is still unopened. The corresponding MLflow run is named `gaussian-joint-validation-region-audit-seed17`.

Review scores assess the plans, not training results or evidence of chip compatibility. Preserve the original protocols and previously used test sets; read the plan before running experiments. The legacy source remains at `../../word2energy` while archive recovery is being verified. The direct tracking dependency is listed in `requirements-tracking.txt`; the tested environment freeze and local database are in `tracking/` and must be backed up rather than committed.
