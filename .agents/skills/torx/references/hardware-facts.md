# Hardware facts with sources

Tag legend: **[WP]** whitepaper arXiv:2608.01612v2 (page in wiki/paper/page), **[BLOG]** extropic.ai "From One to One Billion" (pasted by the user, not independently checked), **[REPO]** torx source, **[OWN]** our run, **[ASSUME]** assumption, not sourced.

## Z1

| Fact | Tag |
|---|---|
| Hardwired sparse degree-16 interaction graph, each pbit coupled to 16 neighbours via on-die wiring beyond nearest neighbour | [WP p9 Fig. 7] |
| Chromatic Gibbs sweeps without FPGA; host sends biases `h_i` and couplings `J_ij`, chip returns samples | [WP p9-10] |
| Evaluates **2-local Glauber generators** natively; the logical PSC must be compiled to a hardware-compatible form (companion paper arXiv:2608.01615, Thermalizers) | [WP p10] |
| 269,568 pbits, 8 cores, 215,904 coupling parameters, >50 MHz, <1 W, <12 mm per side | [BLOG] |
| Early access 2027; tapeout done | [BLOG, stack figure] |
| 215,904 coupling parameters vs about 2.16 M edges (`269,568*16/2`): per-edge programmability unknown | [OWN arithmetic] open question |
| No on-chip adder, accumulator or counter is documented | [WP, searched] |
| Bias DAC resolution on Z1 | **unknown** |

## X0 / XTR-0 (test chip and desktop platform)

| Fact | Tag |
|---|---|
| Control through on-chip 8-bit DACs; weak subthreshold outputs amplified off-chip and digitised on the daughterboard | [WP p20 App. D] |
| pbit calibrated by Bayesian optimisation; per-site calibration error about 5% absolute probability | [WP p9] |
| Relaxation time `tau0 ~ 100 ns` (autocorrelation `exp(-tau/tau0)`); designs span 1 ns to 1 ms; observe at intervals much longer than `tau0` for near-independent draws | [WP p20] |
| Energy per independent sample about 15 aJ for a Gaussian primitive; hundreds of attojoules to single femtojoules per sample at tens of MHz for pbits (quoted) | [WP p9, p20] |
| Primitives: pbit (Bernoulli), pdit (categorical), pmode (Gaussian 1D/2D), pMoG (mixture) | [WP p21 Table II] |
| Hardware randomness gave Metropolis-Hastings TV 0.1126 (software 0.0887) and importance sampling TV 0.0622 (software 0.0525) on an 8-site ring, 20,000 samples | [WP p14] |

Implications for emulation: `cookbook_8`/`cookbook_10` model bias quantisation (generic n bits and range) and correlated samples (generic AR(1) and thinning); the roughly 5% calibration error and the actual X0 `tau0` are not modelled, and no Z1 bias range is known.

## Torx backends actually in the repo

`StateVectorSimulator`, `BranchingSimulator`, `AffineGaussianSimulator` [REPO]. The paper names `SampleSimulator` and `HybridSampleSimulator`; those names do not exist in this source [REPO grep]. There is no hardware backend in the open-source repo.
