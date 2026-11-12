# THRML versus Torx: which to use

| Need | Use | Why |
|---|---|---|
| Sample or train an Ising/Boltzmann model with many spins, block Gibbs, contrastive divergence | **THRML** (`thrml` skill) | `IsingEBM`, graph colouring, `estimate_kl_grad`; scales to thousands of spins |
| Program of kernels with depth: diffusion steps, regime switching, graph networks, mixed discrete/continuous | **Torx** | gates/factors compose in a DAG, trainable via expectations |
| Exact small-circuit reference, gradient check | Torx `StateVectorSimulator` | exact, differentiable |
| Hardware-near p-bit experiments for Z1 | THRML (Z1 is Ising Gibbs) | Torx gates reach hardware through Thermalizers (not released) |
| Independent p-bits with chosen means | either; THRML needs a dummy weight-0 edge (empty edge list raises `IndexError`), Torx `PNOT` has no such quirk | |

Relation: Torx programs compile (Thermalizers, [WP p14], paper arXiv:2608.01615) into THRML energy-based models. Notebook 16 itself notes Gibbs is "not Torx's main use case; THRML is the dedicated library".

## Worked hybrid: same estimator in both

`examples/torx_learn.py` section 4 builds the p-bit dot product with 576 `PNOT` gates and `BranchingSimulator`; `examples/z1_attention.py` builds it with a THRML `IsingEBM` (576 independent spins, one weight-0 edge, two blocks). Both give an error falling as `1/sqrt(N)`: Torx RMS error 124 -> 3.3 (N=10 to 10000), THRML method B single-run `B |err|` 56 -> 2.4 (N=10 to 10000); one is an RMS over repeats and the other a single run, so compare the trend, not the numbers. For anything with couplings between spins use THRML; for programs of conditional kernels use Torx.

Run side by side (`cookbook_12_study.py` part 6, d=288, N=1000, 20 repeats, exact `q.k` 143.36): Torx 576 `PNOT` gates: mean 142.24, sd 5.04; THRML 576 independent spins: mean 142.34, sd 6.64 (formula 5.67; 20 repeats give a wide interval on an sd). A Welch test on the two sets of 20 means gives t = -0.05, p = 0.96 and a variance ratio Torx/THRML of 0.58: no evidence of a difference, but with 20 repeats each this fails to reject equality; it does not prove equivalence (the independent-spin law is the same by construction).

## Gradients across the two

THRML: contrastive moment differences (`estimate_kl_grad`); never differentiate the sampler. Torx: expectations via parameter shift, or score-function terms you write (gradients.md).
