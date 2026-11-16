# Independent critic: fresh synthetic corpus 227

16 November 2026. GPT-6.1 Sol Ultra examined the completed corpus 227 runs
while the frozen 72-run panel continued. This is a post-hoc diagnostic and does
not change the registered endpoint or gate.

At both lexical400 and joint100, random passed 2/4 model seeds, spectral 4/4,
and the fixed row-permuted spectral control 4/4. All 12 runs had lexical AUC
1.0. Mean joint100 context gain was 0.222595, 0.445286, and 0.467440,
respectively. The spectral-minus-permuted success-rate difference is zero, so
the prespecified requirement of a strict advantage on every corpus has failed.

The critic inspected all 12 configurations, source snapshots, initial hashes,
permutation, paired internal/interaction initialization, protocol hash and
fixed handoff. Independent NumPy enumeration of the 24 lexical400 and joint100
endpoints reproduced sense, AUC, gain, anchor/context TV and joint loss with
maximum absolute discrepancy 5.55e-16. This audit is scoped to corpus 227.

The permutation genuinely altered initial alignment: the fraction of
anchor-bias variance between the three families dropped from 0.9521 to 0.05316,
and the F–R centroid distance from 0.2480 to 0.07810. Initial sense was about
0.917–1.0 for spectral and 0.5–0.667 for permuted; by lexical100 both reached
1.0 for all four model seeds. These trajectory observations are exploratory.

The oracle-assisted negative sampler gives lexical learning information about
family separation: in this corpus, 134 of 320 F-anchor negative draws were
R-anchors, and 137 of 320 R-anchor negative draws were F-anchors. That is
consistent with training repairing the shuffled arrangement; it is not proof
of the sole cause. This known oracle limitation prevents interpreting the
result as unsupervised language learning. One fixed permutation cannot tell
whether the spectral-shaped distribution itself helps optimization or whether
this permutation is unusually favorable. Four model seeds do not constitute
four independent permutations.

The critic proposed an exploratory panel on this already-observed corpus: eight
new frozen permutations by four model seeds, keeping the original budget and
endpoints, with permutations as units. Such a panel would investigate the
mechanism, not reverse the failed gate or establish natural-language or
hardware performance. It was not run as part of the frozen panel.
