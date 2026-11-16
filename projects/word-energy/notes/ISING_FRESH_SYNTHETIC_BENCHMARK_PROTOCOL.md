# Frozen protocol: fresh synthetic corpus replication

16 November 2026. Prospective **development benchmark**, specified before any
model results; critic GPT-6.1 Sol Ultra reviewed the design. The original
84-window corpus has exhausted its 28 anchor combinations per family, so
repartitioning its old tokens cannot create genuinely new examples.

## Question and limits

Does train-cooccurrence spectral bias initialization produce more consistent
toy validation performance than random initialization and a fixed row-shuffled
spectral control over independently generated synthetic corpora?

Six distinct corpus namespaces/replicates `211,223,227,229,233,239`; each has
24 new opaque anchor tokens and the same center tokens `bank`/`song`. Family
membership is assigned by a deterministic shuffle; vocabulary order does not
encode family. Independent per-family RNG streams generate 16/6/6 train/
validation/reserved-test unordered anchor combinations, accepting partitions
only if each train anchor has degree >=2. No data draw, eigengap failure, or
seed may be discarded based on measured model performance. Different corpus
replicates share a grammar and center types, but not anchor identities; they
are synthetic relational instances, **not independent natural-language
documents**. Anchors occur in training, so validation tests new combinations,
not unseen word-type generalization.

Four paired model seeds `251,257,263,269` per corpus and three arms: random,
spectral, row-permuted spectral with RNG6061. Because vocabulary sizes match,
this yields one identical numeric permutation for all six corpora; it is not
six independent permutation draws. Initial
internal/interaction parameters match within each paired seed. Spectral receives
only that corpus's training token IDs and vocabulary size. Corrected F3, frozen
400 lexical + 100 joint updates, fixed lexical400→joint100 handoff, original
optimizer/negative schedule, no test evaluation and no posthoc tuning.
Original synthetic negatives/corruptions still use family/grammar oracle;
sense map uses training labels. Neither the pipeline nor benchmark is a
completely unsupervised real-language test.

## Measurements and decisions

Per run, record lexical400 and joint100 sense, per-family counts, AUC, gain,
anchor/context TV, hashes, source snapshot, seed, split, runtime, peak working
set, failures, and handoff. Primary per-run indicator at fixed joint100:
sense >=10/12, lexical AUC >=0.75, mean context gain >=0.03. Lexical400 is a
required companion outcome. A spectral eigengap failure counts as method
failure for spectral and permuted arms, and is recorded without redrawing the
corpus; random control must remain runnable.

Primary unit is corpus replicate (n=6), not 24 seed runs. For each corpus and
arm, average four paired-seed success indicators. Compare spectral minus random
and spectral minus permuted, separately. Conservative promotion criterion:
both differences strictly positive in all six replicates; ties fail promotion.
Under independent, symmetric split-level null signs, each one-sided exact
sign probability is 1/64; Bonferroni-adjusted probability for two comparisons
is 0.03125. The assumption and n=6 limit inference to this synthetic generator,
not broad model superiority. Report all replicate-level rates, gains and
failures even if criterion fails. No early stop on model performance.

## Budget, provenance and stop rules

72 sequential runs × 500 optimizer updates = 36,000 maximum updates; target
60-minute batch wall time, process peak working set <=2 GiB, OMP/OPENBLAS
environment set to four threads. Resource checks occur at snapshots; these are
not hard OS caps. Stop on correctness/provenance failure, nonfinite state or
measured resource overrun; an unfinished batch is inconclusive. Individual
failures remain in the ledger; never silently replace a run. Before training,
verify v2 injection through training, diagnostics, scoring with legacy data
access patched to raise, disjoint splits, coverage, replayable seeds and legacy
protocol hash. Store exact source, dataset generation version, token map,
training token/matrix/permutation hashes, config and checkpoint snapshot per
run. Additional spectrum failure has explicit handling; never substitute a
different split to achieve a numerically favorable result.
