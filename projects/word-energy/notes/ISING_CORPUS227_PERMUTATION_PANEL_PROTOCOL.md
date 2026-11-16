# Protocol: corpus 227 permutation follow-up

16 November 2026. This exploratory follow-up was chosen after observing corpus
227 tie in the frozen six-corpus benchmark. It cannot alter that failed gate.
Question: are many different row permutations of the same spectral bias cloud
able to pass the same fixed-budget endpoint on this particular observed corpus,
or was the previously observed RNG6061 arrangement exceptional?

Freeze eight new NumPy `default_rng` permutation seeds: 7011, 7013, 7019,
7027, 7039, 7043, 7057, 7069. Permute all vocabulary rows, as in the original
control; verify unique permutations and that none equals RNG6061. Evaluate each
on model seeds 251, 257, 263, 269, paired to the existing corpus 227 spectral,
random and RNG6061 results. That is 32 new runs, no redraw or result-based
selection. The spectral cloud, corpus, oracle-assisted training schedule,
corrected F3, lexical400 to joint100 handoff and joint100 thresholds are the
same as the original panel. Track lexical400 as well. No test scoring.

The unit for the new variation is permutation, not the 32 individual runs
or new datasets. Report all eight 0–4/4 pass counts and per-run endpoints,
along with the original controls and initial alignment where feasible. No
confirmatory p-value, promotion claim, or inference to other corpora follows
from this selected corpus. If many permutations pass, alignment-specific
necessity is weakened on this setup; if few pass, RNG6061 may be favorable.
Neither outcome by itself proves a causal mechanism.

Use local classical exact enumeration and MLflow experiment
`word-energy-corpus227-permutation-panel-v1`, 2 GiB per-process working-set
checks, OMP/OPENBLAS four-thread environment, and 40-minute batch deadline.
Failure or timeout is preserved and the panel is reported incomplete; do not
silently replace runs. Compare source and data provenance with saved prior
controls before starting. No THRML or physical-chip performance is measured.
