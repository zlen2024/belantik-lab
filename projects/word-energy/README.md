# Word Energy research

This is the active workspace for Word Energy Embedding. It is currently a scaffold: the new model has not yet been implemented or trained.

- [Primary Ising/Z1 plan](plans/word2energy/WORD_ENERGY_Z1_FIRST_PLAN.md)
- [Gaussian baseline plan](plans/word2energy/WORD_ENERGY_EMBEDDING_V2_PLAN.md)
- [Original ideas and conversation](plans/word2energy/RUMUSAN_IDEA_BIKAR_ENERGY_EMBEDDING.md)
- [Ising review](plans/word2energy/reviews/z1_first/REVIEW_LOG.md) and [Gaussian review](plans/word2energy/reviews/energy_embedding_v2/REVIEW_LOG.md)
- [Experiment tracking decision](TRACKING_DECISION.md): the user selected local MLflow.

Review scores assess the plans, not training results or evidence of chip compatibility. Preserve the original protocols and previously used test sets; read the plan before running experiments. The legacy source remains at `../../word2energy` while archive recovery is being verified. The direct tracking dependency is listed in `requirements-tracking.txt`; the tested environment freeze and local database are in `tracking/` and must be backed up rather than committed.
