# From paper to a scoped implementation

Start with the requested algorithm or component. Inspect official author code when available and relevant; record its revision and differences from the paper. Do not silently substitute a third-party implementation or launch a full training run.

## Map method to code

For substantial implementations, keep a compact mapping alongside the code:

| Paper section / equation | Operation | Code location | Assumption or deviation | Check |
| --- | --- | --- | --- | --- |

Resolve the objective, tensor shapes, initialization, preprocessing, sampling schedule, update rule, stopping criterion and evaluation protocol. Missing hyperparameters are assumptions to record, not values to attribute to the paper. Choose documented reasonable defaults when they do not change the method; ask only when a missing detail prevents a meaningful implementation.

## Implement and verify

- Use a small deterministic case to check shapes, signs, indexing, units and boundary behavior before scaling up.
- Compare against an analytical or enumerated solution when practical. Test scientific invariants and meaningful outcomes rather than tests that repeat implementation formulas.
- Record the dependency/runtime versions, exact command, seed, inputs and relevant outputs of actual verification. Distinguish implemented, smoke-tested, evaluated and reproduced.
- Reproduction claims need a comparable data/evaluation protocol and an explicit tolerance or uncertainty analysis. A successful import or plausible learning curve is insufficient.

## THRML and thermodynamic models

Read the installed THRML skill and inspect the local library/API before implementation. Make the following assumptions explicit in the mapping:

- State encoding (binary versus spin), energy sign, coupling counting, temperature or inverse temperature, and any conversion of biases or weights.
- Which variables are clamped, which are free, and which nodes can update independently within each block.
- Positive/negative phase definitions, initialization, burn-in, retained samples, chain count and convergence/mixing evidence.
- Whether the target method requires discrete block Gibbs sampling or another dynamics. THRML compatibility is a conclusion to justify.

For a sufficiently small model, compare sampled probabilities or moments with exact enumeration under the same energy convention, allowing sampling uncertainty. Check learning-update signs separately. Do not differentiate through discrete Gibbs draws as though they were a reparameterized sampler.

Separate simulator measurements from hardware measurements and theoretical complexity. Do not claim hardware latency or energy benefits from a JAX simulation alone.
