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

Current status: three exact-gradient synthetic prechecks (seeds 17, 29 and 43) have been recorded locally. Seeds 17 and 43 reached the toy sense/retrieval gate, while seed 29 did not; therefore the three-seed Ising pilot does **not** pass its preregistered gate. Baselines, the lexical-positive permutation control, sampler audit and physical-hardware mapping are still outstanding. The local MLflow experiment `word-energy-exact-training` is the record of individual runs.

Review scores assess the plans, not training results or evidence of chip compatibility. Preserve the original protocols and previously used test sets; read the plan before running experiments. The legacy source remains at `../../word2energy` while archive recovery is being verified. The direct tracking dependency is listed in `requirements-tracking.txt`; the tested environment freeze and local database are in `tracking/` and must be backed up rather than committed.
