"""Cookbook 2: hybrid circuit and Gaussian simulator. Run from OUTSIDE master/."""
import jax, jax.numpy as jnp, numpy as np
from torx import psc

# ---- 1. regime switching: PditCycle (discrete) then MixtureGaussianGate (continuous) ------
# MixtureGaussianGate: x' = x + means[k] + N(0, exp(log_vars[k])), k = value of the control pdit.
# sites = (discrete_site, continuous_site); dims = continuous dims only.
gates = [psc.PditCycle(0, dims=3),
         psc.MixtureGaussianGate(sites=(0, 0), dims=(1,), num_components=3)]
circ = psc.HybridPCircuit(gates)
params = circ.init_params(jax.random.key(0))
print("param shapes:", [jax.tree.map(lambda a: a.shape, p) for p in params])
means = jnp.array([[2.0], [0.0], [-2.0]])
params = [jnp.array([2.0, -2.0]),                       # PditCycle: strongly forward
          {"means": means, "log_vars": jnp.full((3, 1), -2.0)}]
state = {"discrete": jnp.array([0], dtype=jnp.int32), "continuous": jnp.zeros(1)}
out = circ.sample(jax.random.key(1), state, params)
print("hybrid sample ->", out)
outs = circ.sample_multiple(jax.random.key(2), state, params, n_samples=200000)
d, x = np.array(outs["discrete"][:, 0]), np.array(outs["continuous"][:, 0])
p_fwd = np.exp(2.0) / (1 + np.exp(2.0) + np.exp(-2.0))
print(f"P(state=1) sampled {np.mean(d == 1):.4f} (SE {np.sqrt(p_fwd * (1 - p_fwd) / len(d)):.4f}, {len(d)} samples) vs softmax {p_fwd:.4f};  "
      f"mean x given state 1: {x[d == 1].mean():+.3f} (expect {float(means[1, 0]):+.1f})")

# ---- 2. Gaussian simulator: exact moments, no sampling ------------------------------------
# AffineGaussianGate: x' = A x + b + N(0, exp(log_var)); theta = dict(A, b, log_var)
# Gate 1 draws x0 ~ N(0,1) from a point mass; gate 2 sets x1 = 2*x0 + N(0, 0.25) (x0 kept).
# The input to the simulator is a POINT MASS, so correlation must be created by gates.
g = [psc.AffineGaussianGate(sites=[0], dims=(1,)),
     psc.AffineGaussianGate(sites=[0, 1], dims=(1, 1))]
gc = psc.HybridPCircuit(g)
th = [{"A": jnp.eye(1), "b": jnp.zeros(1), "log_var": jnp.zeros(1)},
      {"A": jnp.array([[1.0, 0.0], [2.0, 0.0]]), "b": jnp.zeros(2),
       "log_var": jnp.log(jnp.array([1e-6, 0.25]))}]
gs = psc.AffineGaussianSimulator()
comp = gs.build_circuit(gc, th)
mom = gs.propagate(comp, jnp.zeros(2))
print("mean", np.array(mom.mean), "\ncov\n", np.array(mom.covariance).round(3))
# analytic: Var x0 = 1, Cov(x0,x1) = 2, Var x1 = 4 + 0.25
cond = gs.condition(comp, {1: jnp.array([2.0])}, initial_continuous=jnp.zeros(2))
print("posterior mean given x1 = 2:", np.array(cond.mean).round(3),
      " analytic E[x0|x1=2] =", round(2 * 2.0 / 4.25, 3))

print("softmax([0, 2, -2]) by hand:", np.round(np.array(jax.nn.softmax(jnp.array([0.0, 2.0, -2.0]))), 3))
