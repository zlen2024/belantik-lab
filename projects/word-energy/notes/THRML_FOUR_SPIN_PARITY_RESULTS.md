# THRML four-spin simulator parity audit

Date: 2026-11-19. MLflow run `ce2be3890c8e43bc9e74b4897feedc87`; frozen protocol and code commit `4886116`; source model run `457aaa1ab2af46678844386986a61b32`, seed 17 epoch 3. Full exact probabilities, observables and raw sample chains are stored in MLflow.

## Purpose and method

Check whether a learned joint four-spin model can be represented by a THRML Ising graph and whether simulated Gibbs samples agree with exact enumeration for two illustrative triples. CPU simulator only. Graph: 12 spins, 26 edges, maximum degree 5, four valid update colors; beta 1. Per triple and sampler seed: four dispersed chains, 500 warmup and 8,000 retained draws per chain. Compare 12 spin probabilities and eight cross-edge correlations to exact 4,096-state probabilities.

Predefined gate for every observable: R-hat <1.05, ESS >=400, MCSE <=0.01 and absolute error <=max(0.02, 3 MCSE). A constant sampled observable makes R-hat/MCSE undefined and fails this gate. TV distance and independent-sample TV are descriptive and do not prove mixing.

| Fixture | Seed | Full-state TV | IID TV mean | Failed observables | Gate |
|---|---:|---:|---:|---|---|
| finance | 17 | 0.01897 | 0.01386 | edge2_6, edge3_7 | FAIL |
| finance | 29 | 0.01553 | 0.01386 | edge1_5, edge2_6, edge3_7 | FAIL |
| finance | 43 | 0.01513 | 0.01386 | edge2_6, edge3_7 | FAIL |
| river | 17 | 0.00325 | 0.00345 | spin0_plus | FAIL |
| river | 29 | 0.00390 | 0.00345 | spin0_plus, spin3_plus | FAIL |
| river | 43 | 0.00297 | 0.00345 | spin0_plus | FAIL |

Exact enumerated normalized scores agree with production calculation within 1e-11 before sampling. All six sampling gates failed: in river, very rare spin flips were not observed in some chains, leaving convergence statistics undefined; in finance, some cross-edge correlations had MCSE above the prespecified 0.01 threshold. Small full-state TV does not override these failed diagnostics. No retry or gate change was performed.

This is a simulator mapping and a limited diagnostic on two selected triples, not proof of robust sampling across text, free-energy estimation, hardware deployment, or power/speed advantage on a chip. A separate protocol must examine rare states and sampling precision before another run; keep this initial failure visible.

Independent Sol 6.1 ultra review found 109/120 observable checks passed. Four failures were constant rare-event river indicators; seven were finance cross-edge MCSE above the threshold (0.01045–0.01213). All nonconstant observables met the R-hat, ESS and exact-error conditions. River spin 0 has exact minority probability about 1.267e-5: only 0.406 expected minority observations in 32,000 independent draws, so a constant sample alone is not proof of trapping. These observations diagnose the failures but do not reverse the failed gate.

The critic also observed uneven finance rare-event counts: spin 0 has exact minority probability 0.01029 (about 329 expected in 32,000 draws) versus 100, 428 and 222 observed across sampler seeds. Correlated samples and a permissive absolute-error threshold mean passing individual checks does not establish rare-mode weights. Longer-chain work requires a separately frozen protocol and must not be presented as a rescue of this failed run.
