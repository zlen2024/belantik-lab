"""Cookbook 5: chromatic Gibbs on an Ising ring with Torx gates (worked example for explanations).

Paper Eq. (58)-(60): local field l_i = h_i + sum_j J_ij (2 sigma_j - 1), P(sigma_i = 1 | rest) = sigmoid(2 beta l_i),
one site update = PNOT(theta = 2 beta l_i) after PReset(inf).  Step 1 checks that identity on the exact
state-vector simulator. Step 2 runs the full chromatic sampler and compares with the exact Boltzmann law,
reporting the finite-sample TV floor so the result can be read honestly. Run from OUTSIDE master/.
"""
import itertools
import jax, jax.numpy as jnp, numpy as np
from torx import psc

# ---- 1. one-site kernel: PReset(theta=+20 ~ infinity) then PNOT(2*beta*l) ---------------------
beta, l = 1.0, 0.7
circ = psc.DiscretePCircuit([psc.PReset(0), psc.PNOT(0)])
sv = psc.StateVectorSimulator()
for start in (0, 1):
    x = jnp.zeros(2).at[start].set(1.0)
    dens = sv.density(sv.build_circuit(circ, [jnp.array([20.0]), jnp.array([2 * beta * l])]), x)
    print(f"start {start}: P(1) = {float(dens[1]):.4f}   sigmoid(2*beta*l) = {float(jax.nn.sigmoid(2*beta*l)):.4f}")

# ---- 2. chromatic Gibbs on an 8-site ring, compared with exact Boltzmann -----------------------
n, beta = 8, 1.5
rng = np.random.default_rng(0)
J = rng.normal(0, 0.6, n)           # J[i] couples i and i+1 (mod n)
h = rng.normal(0, 0.3, n)
configs = np.array(list(itertools.product([-1, 1], repeat=n)))
E = -(sum(J[i] * configs[:, i] * configs[:, (i + 1) % n] for i in range(n)) + configs @ h)
P = np.exp(-beta * E); P /= P.sum()                      # exact distribution over 2^8 states

def field(s):
    return h + J * np.roll(s, -1, axis=1) + np.roll(J, 1) * np.roll(s, 1, axis=1)

chains, sweeps = 6000, 240
s = rng.choice([-1, 1], size=(chains, n))
for _ in range(sweeps):
    for color in (np.arange(0, n, 2), np.arange(1, n, 2)):    # two-colourable ring
        p1 = 1 / (1 + np.exp(-2 * beta * field(s)[:, color]))
        s[:, color] = np.where(rng.random(p1.shape) < p1, 1, -1)
idx = ((s + 1) // 2) @ (2 ** np.arange(n - 1, -1, -1))
cidx = ((configs + 1) // 2) @ (2 ** np.arange(n - 1, -1, -1))
emp = np.bincount(idx, minlength=2 ** n) / chains
order = np.argsort(cidx)
tv = 0.5 * np.abs(emp - P[order]).sum()
floor = np.mean([0.5 * np.abs(rng.multinomial(chains, P[order]) / chains - P[order]).sum() for _ in range(30)])
print(f"\nTV(sampler, exact) = {tv:.4f};  finite-sample floor for an EXACT sampler with {chains} draws = {floor:.4f}")
print("=> TV is statistically consistent with zero bias if it is near the floor, not far above it.")
