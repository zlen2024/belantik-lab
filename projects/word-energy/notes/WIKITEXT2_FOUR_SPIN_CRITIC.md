# Independent critic: four-spin experiment

Critic: GPT-6.1-Sol, ultra effort. Independent read-only numerical audit; date 2026-11-19.

- Verified 12 new runs completed and checked hashes for 96 checkpoints, 15 dataset arrays and all 12 identical frozen-source snapshots.
- Independently recomputed validation scores/objectives using NumPy 16-state messages: maximum discrepancy 1.33e-15; 32 direct state calculations per Ising checkpoint: maximum discrepancy 1.78e-15. Recomputed landscape diagnostics with maximum discrepancy 3.55e-15.
- Independently recomputed WiC predictions with log-message posteriors/cosines (max 3.33e-16), direct enumeration up to 20 spins (max 1.11e-16), AUC and clustered bootstraps.
- Joint4 improves the prior joint2 development panel, but changes multiple capacity/coupling factors; it trails the parameter-matched vector and has no clear macro advantage over pair_add4.
- Most initialized words and an even larger fraction of shuffled words have multiple local minima, showing that minima count alone is not evidence of semantic senses.
- WiC joint4 AUC is 0.52994 versus 0.59149 at initialization, 0.53711 shuffled, and 0.58508 for vector10. Small reused subset and wide intervals limit the conclusion; the run does not demonstrate a sense gain.

THRML sampling parity did not meet its previously specified precision/convergence gate. See the companion parity report; preserve the original run and investigate failures before any new fixed protocol.

Overall: four spins are a feasible exact small-system software model, not a demonstrated semantic or thermodynamic hardware improvement.

THRML diagnosis: 109/120 checks passed; four rare river indicator checks were constant and seven finance cross-edge checks missed the MCSE threshold. All nonconstant checks met R-hat, ESS and exact-error conditions. This suggests precision and rare-event limitations, not a demonstrated sign or graph-coloring error.
