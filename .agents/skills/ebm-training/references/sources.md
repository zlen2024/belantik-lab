# Sources: papers and code

Status column: **PDF** = read in this session; **web** = existence and gist confirmed by search this session (not read in full); **recall** = from memory, not verified here. Re-verify "recall" rows before citing. Search date 2026-11-08.

## A. Core papers

| Work | Why | Status |
|---|---|---|
| Song & Kingma, *How to Train Your Energy-Based Models*, arXiv:2101.03288 | base tutorial (MLE+MCMC, SM/DSM/SSM, NCE, Stein, adversarial) | PDF |
| LeCun, Chopra, Hadsell, Ranzato, Huang, *A Tutorial on Energy-Based Learning* (2006, MIT Press chapter) - [PDF](https://cs.nyu.edu/~yann/2011f-G22-2565-001/diglib/lecun-06.pdf) | energy vs loss functional, margin losses, collapse | **PDF** (sections 2.1-2.2 and Eq. 6-25 read and checked) |
| LeCun, *A Path Towards Autonomous Machine Intelligence* (2022, OpenReview BZ5a1r-kVsf) | JEPA, regularised vs contrastive, VICReg | web only: the OpenReview PDF could not be retrieved here (HTML returned), so section 4 of `lecun-framework.md` stays [recall] |
| Hinton, *Training products of experts by minimizing contrastive divergence* (2002) | CD | cited in PDF |
| Tieleman, *Training RBMs using approximations to the likelihood gradient* (ICML 2008) | PCD | cited in PDF |
| Hinton, *A Practical Guide to Training Restricted Boltzmann Machines* (2012) | RBM engineering practice | web (existence) |
| Salakhutdinov & Murray, *On the quantitative analysis of deep belief networks* (ICML 2008) - [page](https://homepages.inf.ed.ac.uk/imurray2/pub/08dbn_ais/) | AIS estimate of RBM `Z` | web |
| Ackley, Hinton, Sejnowski, *A learning algorithm for Boltzmann machines* (1985) | origin of the two-phase rule | recall |
| Hyvärinen (2005); Vincent (2011); Song et al. (2019); Song & Ermon (2019, 2020); Song et al. (2021) | SM, DSM, SSM, NCSN, score SDE | cited in PDF |
| Gutmann & Hyvärinen (2010); Gutmann & Hirayama (2012) | NCE and Bregman generalisation | cited in PDF |
| Du & Mordatch, *Implicit Generation and Generalization in EBMs*, arXiv:1903.08689 | replay buffer, L2 energy reg., spectral norm, short-run Langevin at scale | web |
| Nijkamp, Hill, Han, Zhu, Wu, *On the Anatomy of MCMC-Based Maximum Likelihood Learning of EBMs*, arXiv:1903.12370 (AAAI 2020) | convergent vs non-convergent MCMC; short-run models are generators | web |
| Du, Li, Tenenbaum, Mordatch, *Improved Contrastive Divergence Training of EBMs*, arXiv:2012.01316 (ICML 2021) | the often-dropped gradient term of CD | web |
| Grathwohl et al., *Your Classifier is Secretly an Energy Based Model*, arXiv:1912.03263 (ICLR 2020) | JEM, SGLD with re-initialisation | web |
| Grathwohl et al., *Learning the Stein Discrepancy for Training and Evaluating EBMs without Sampling* (ICML 2020), arXiv:2002.05616 | learned Stein critic | web |
| Grathwohl et al., *Oops I Took A Gradient: Scalable Sampling for Discrete Distributions*, arXiv:2102.04509 | Gibbs-with-gradients sampler for discrete EBMs | web |
| Welling & Teh, *Bayesian learning via stochastic gradient Langevin dynamics* (2011) | SGLD | recall |
| Neal, *Annealed importance sampling* (2001) | AIS | cited in web result |
| Salakhutdinov & Hinton, *Deep Boltzmann Machines* (2009) | multi-layer undirected models | recall |
| Gao et al., *Flow Contrastive Estimation of EBMs* (CVPR 2020); Gao et al., *Learning EBMs by diffusion recovery likelihood* (ICLR 2021) | learned noise, multi-scale recovery | FCE: cited in PDF; recovery likelihood: recall |
| Bardes, Ponce, LeCun, *VICReg*, ICLR 2022 | variance-invariance-covariance regularisation | recall (named in the 2022 paper per web summary) |
| Assran et al., I-JEPA (2023); Bardes et al., V-JEPA (2024) | JEPA instantiations | recall |
| Jelinčič et al. (Extropic), *An efficient probabilistic hardware architecture for diffusion-like models*, arXiv:2510.23972 - [abs](https://arxiv.org/html/2510.23972v2) | DTMs: chains of simple EBMs on sparse Boltzmann machines; the hardware-training context for THRML | web (headline energy numbers differ between versions: check) |
| Whitelam, *Generative thermodynamic computing*, arXiv:2506.15121 | thermodynamic-computing generative modelling | web (existence) |
| Hyvarinen, *Some extensions of score matching* / ratio matching (2007) | discrete-data analogue of SM | cited in PDF (extensions); ratio matching: recall |
| Schroeder et al., *Energy Discrepancies: A Score-Independent Loss for EBMs*, arXiv:2307.06431; *Training Discrete EBMs with Energy Discrepancy*, arXiv:2307.07595 | sampling-free loss, discrete version | web |
| Zhang et al., *A Langevin-like sampler for discrete distributions* (DULA/DMALA, ICML 2022) | gradient-informed discrete MCMC | recall |
| Montavon & Mueller, *Deep Boltzmann machines and the centering trick* (2012); Melchior et al., *How to center deep Boltzmann machines* (JMLR 2016) | centering for BMs | recall |
| Burda, Grosse, Salakhutdinov, *Accurate and conservative estimates of MRF log-likelihood using reverse annealing* (AISTATS 2015) | RAISE | recall |
| Grosse, Ghahramani, Adams, *Sandwiching the marginal likelihood using bidirectional Monte Carlo* (2015) | exact-sample reverse AIS bracket used in `ais_logz.py` | recall |
| Du et al., *Compositional Visual Generation with Energy Based Models* (NeurIPS 2020); Du et al., *Reduce, Reuse, Recycle* (ICML 2023) | composition of energies | recall |
| Swendsen & Wang (1986); Geyer (1991) parallel tempering; Gelman & Rubin / Vehtari et al. 2021 (rank-normalised split R-hat) | replica exchange, convergence diagnostics | recall |
| Salimans & Ho, *Should EBMs model the energy or the score?* | parametrising score vs energy | recall, **not found by search** |

## B. Code

| Repo | Use | Status |
|---|---|---|
| [openai/ebm_code_release](https://github.com/openai/ebm_code_release) | reference for Du & Mordatch (CIFAR-10, ImageNet32/128, dSprites); setup has deprecated download URLs | web (aggregator; not opened) |
| [rosinality/igebm-pytorch](https://github.com/rosinality/igebm-pytorch) | compact PyTorch port of the same method, MIT | web (aggregator) |
| [wgrathwohl/JEM](https://github.com/wgrathwohl/JEM) | official JEM (training, eval, adversarial attacks) | web |
| [ermongroup/sliced_score_matching](https://github.com/ermongroup/sliced_score_matching) | official SSM (GPL-3.0) | web |
| [yang-song/score_sde](https://github.com/yang-song/score_sde) (JAX) and `score_sde_pytorch` | official score SDE | web |
| [phlippe/uvadlc_notebooks](https://github.com/phlippe/uvadlc_notebooks) Tutorial 8 *Deep Energy-Based Generative Models* ([docs](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html)) | pedagogical CD with buffer, MIT | web |
| [soran-ghaderi/torchebm](https://github.com/soran-ghaderi/torchebm) (PyPI `torchebm`) | PyTorch library: samplers (Langevin, HMC), CD and score-matching losses; API changed across 0.3-0.5 | web (PyPI/aggregator) |
| [extropic-ai/thrml](https://github.com/extropic-ai/thrml), `extro-torx` | THRML / Torx upstream, local copies in `master/thrml`, `master/torx` | in workspace |
| yilundu/improved_contrastive_divergence | official code of Improved CD (project page: energy-based-model.github.io/improved-contrastive-divergence) | **repo URL not confirmed** |
| ermongroup/ncsn, ncsnv2 | NCSN code | recall |
| wgrathwohl `GWG_release` | Gibbs-with-gradients | recall, **not found** |
| facebookresearch/vicreg, ijepa, jepa | LeCun-group JEPA code | recall |

## C. Books / lectures

- Goodfellow, Bengio, Courville, *Deep Learning*, ch. 16 and 18 (structured probabilistic models; confronting the partition function) [recall].
- Murphy, *Probabilistic Machine Learning: Advanced Topics* (energy-based models chapter) [recall].
- LeCun's NYU lecture slides on energy-based learning: [NIPS 2006 tutorial](https://cs.nyu.edu/~yann/talks/lecun-tutorial-nips-2006.pdf), [MLSS 2008](https://cs.nyu.edu/~yann/talks/lecun-20080905-mlss-ebm.pdf) [web].

## D. Reading order for a newcomer

1. Song & Kingma (this skill's `theory-map.md`) -> 2. LeCun 2006 (loss functionals) -> 3. Hinton 2012 + a tiny RBM (`thrml_rbm_negative_phase.py`) -> 4. Du & Mordatch (stability) -> 5. Nijkamp 2019 (what your sampler really trains) -> 6. Score SDE (modern alternative) -> 7. LeCun 2022 (regularised/JEPA direction) -> 8. DTM (hardware).
