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
- `scripts/train_gaussian_joint.py --seed 17`: D=1/M=2 differentiable Gauss--Hermite joint-ranking precheck. It selects a numerically eligible validation checkpoint by macro sense accuracy, then total joint loss, then earliest step; it records adaptive sense-region, domain and smooth free-energy/gradient audits. It does not yet perform sampling.

Current status: three exact-gradient synthetic prechecks (seeds 17, 29 and 43) have been recorded locally. Seeds 17 and 43 reached the toy sense/retrieval gate, while seed 29 did not; therefore the three-seed Ising pilot does **not** pass its preregistered gate. The matching untrained random control exposed a further weakness of the small sense metric: random seed 17 already achieved 10/12 sense accuracy while its lexical AUC was only 0.3125. Thus the readout must be compared with the controls, not interpreted from the threshold alone. The frozen lexical-positive permutation control retained the original joint supervision as specified: seed 17 remained at 1.0 sense accuracy, while seeds 29 and 43 were 0.5. This isolates neither the whole supervision nor a hardware mechanism; it only tests the added lexical-positive contribution. Sampler audit and physical-hardware mapping are still outstanding. The local MLflow experiment `word-energy-exact-training` is the record of individual runs.

The lexical Gaussian baseline reached a holdout retrieval AUC of 1.0 for each of the same three seeds. This does not compare contextual sense handling: it has no Gaussian joint system yet. Its MLflow experiment is `word-energy-gaussian-lexical`.

The Gaussian joint lane has completed the validation-only D=1/M=2 precheck for seeds 17, 29 and 43. A repair removed false sense-region boundaries caused by absolute density tolerances in tails; all selected M2 joint and frozen-lexical checkpoints now pass the numerical audit, and both achieve validation sense accuracy `1.0` for all three seeds. The corrected contextual holdout is frozen in [M2_CONTEXTUAL_HOLDOUT_MANIFEST.md](plans/word2energy/M2_CONTEXTUAL_HOLDOUT_MANIFEST.md) before evaluation.

The frozen M2 contextual holdout also reaches macro sense accuracy `1.0` for every joint and frozen-lexical seed, with exact counts 6/6 in each F and R family and lexical retrieval AUC `1.0`. Joint adaptation increases correct-region posterior gain only by about `0.00364`, `0.00419` and `0.00350` beyond frozen lexical geometry, with no accuracy improvement. The matched tied-D6 and untied-D3 SGNS contextual-vector controls likewise reach `1.0` in every seed. Therefore this small synthetic toy validates the contextual readout but demonstrates **no advantage** for joint Gaussian energy training over ordinary contextual vectors. It does not establish MCMC mixing, natural-text performance, thermodynamic hardware execution or a generally successful model.

Review scores assess the plans, not training results or evidence of chip compatibility. Preserve the original protocols and previously used test sets; read the plan before running experiments. The legacy source remains at `../../word2energy` while archive recovery is being verified. The direct tracking dependency is listed in `requirements-tracking.txt`; the tested environment freeze and local database are in `tracking/` and must be backed up rather than committed.
