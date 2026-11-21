# Existing THRML samples: active-window R reanalysis

21 November 2026. This is a **posthoc reanalysis** of saved THRML CPU-simulator chains from `ce2be3890c8e43bc9e74b4897feedc87`. No sampler was run again. The checkpoint was the older seed-17 margin-trained Ising model, not the later full-softmax model. The old audit's overall precision/rare-event gate failed for every fixture and seed, and remains failed.

The three-token graph contains target `bank`, one context and one additional active token treated here as the candidate. The scalar `R=(1/4)Σ E[s_target,a s_candidate,a]` is computed from products of spins observed in the **same sweep**. Each triple is a separate star. Two exact references agree to within 1e-12: summation over the saved 4,096-state law and the newer active-correlation implementation. The sample arrays' shape, spin encoding and empirical full-state distributions were checked against the source run.

| Active triple | Exact R | Sample R, sampler seeds 17 / 29 / 43 | MCSE range | Posthoc scalar check |
|---|---:|---|---|---|
| bank / river / water | 0.486058 | 0.486688 / 0.486641 / 0.486984 | 0.000729–0.000747 | 3/3 pass |
| bank / money / financial | 0.032767 | 0.032766 / 0.033609 / 0.033313 | 0.000765–0.000787 | 3/3 pass |

For this scalar, all six calculated diagnostics satisfy the earlier observable thresholds for R-hat, ESS, MCSE and error against exact values. That threshold was applied after the original run, so this is descriptive reanalysis, not a newly prespecified gate. Averaging four edge products can be more precise than checking each edge or rare spin separately. A scalar pass does not resolve the old per-edge MCSE failures or unseen rare spin flips. These two illustrative scores also do not establish semantic relevance, usable ranking across candidates, broader convergence, the later checkpoint, or a physical-chip result.

The existing diagnostic computes MCSE as the raw standard deviation divided by the square root of rank-based ESS. Independent critic sensitivity checks using raw-mean autocorrelation ESS and multiple batch sizes found finance MCSE approximately 0.000731–0.000904 and river approximately 0.000597–0.000799. These posthoc checks support the same narrow scalar conclusion for these chains, while leaving rare-state and full-distribution accuracy unresolved.

Source protocol and code were committed as `5b53e13`. MLflow experiment `word-energy-thrml-four-spin-parity`, run `1e60f084b58e4ca5a9098f13e57fe0c8`, contains per-seed diagnostics, exact values, graph and source checkpoint hashes. Independent critic review follows in a separate note.
