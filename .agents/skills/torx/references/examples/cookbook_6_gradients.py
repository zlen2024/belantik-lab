"""Cookbook 6: gradient depth. Run from OUTSIDE master/.

1. Unbiasedness of the three diff_methods at several theta (incl. +/-5), K=2 and K=3 gates.
2. Variance scaling with N and the finite-N bias of param_shift_filter.
3. Silent zero gradient through sample().
4. A per-gate REINFORCE surrogate (paper Eq. 34) written in plain JAX that reproduces the exact gradient.
"""
import jax, jax.numpy as jnp, numpy as np
from torx import psc

METHODS = ("param_shift_inf", "param_shift_single", "param_shift_filter")
sv = psc.StateVectorSimulator()


def exact_and_sampled(circ, x_sv, x_b, make_thetas, grid, f_idx, N, R, methods=METHODS):
    rows = []
    for t in grid:
        th = make_thetas(t)
        ex = jax.grad(lambda z: sv.expval_all(sv.build_circuit(circ, z), x_sv)[f_idx])(th)
        ex = np.concatenate([np.ravel(np.array(a)) for a in ex])
        for m in methods:
            b = psc.BranchingSimulator(num_samples=N, diff_method=m)
            g = jax.jit(jax.grad(lambda z, k: b.expval_all(b.build_circuit(circ, z), x_b, k)[f_idx]))
            try:
                gs = np.array([np.concatenate([np.ravel(np.array(a)) for a in g(th, jax.random.key(i))])
                               for i in range(R)])
            except NotImplementedError:
                rows.append((t, m, None)); continue
            se = gs.std(0, ddof=1) / np.sqrt(R)
            rows.append((t, m, (np.abs(gs.mean(0) - ex).max(), float((se).max()), gs.std(0).max())))
    return rows


# ---- 1. unbiasedness: K=2 chain, theta = [t, t, t] ----------------------------------------
c2 = psc.DiscretePCircuit([psc.PNOT(0), psc.PCNOT([0, 1]), psc.PNOT(1)])
rows = exact_and_sampled(c2, jnp.zeros(4).at[0].set(1.0), jnp.zeros(2, jnp.int32),
                         lambda t: [jnp.array([t])] * 3, (-5.0, -2.0, 0.0, 2.0, 5.0), 1, 4000, 30)
print("K=2 chain, N=4000, 30 repeats:  max|mean-exact|   max SE   max sd")
for t, m, r in rows:
    print(f"  theta={t:+.0f} {m:20s}", "n/a" if r is None else f"{r[0]:.4f}  {r[1]:.4f}  {r[2]:.4f}")

# K=3 pdit: PditCycle theta = [t, -t], loss = mean index of the pdit (expval of a pdit is the mean index)
c3 = psc.DiscretePCircuit([psc.PditCycle(0, dims=3)])
rows = exact_and_sampled(c3, jnp.array([1.0, 0, 0]), jnp.zeros(1, jnp.int32),
                         lambda t: [jnp.array([t, -t])], (-3.0, 0.0, 3.0), 0, 4000, 30)
print("\nK=3 pdit (PditCycle):")
for t, m, r in rows:
    print(f"  theta=[{t:+.0f},{-t:+.0f}] {m:20s}", "raises NotImplementedError (K>2)" if r is None else
          f"{r[0]:.4f}  {r[1]:.4f}  {r[2]:.4f}")

# ---- 2. variance ~ 1/N, and finite-N bias of param_shift_filter -------------------------------
print("\nsd of gradient vs N (theta=0, 3-gate chain), expect ratio ~ sqrt(4000/1000) = 2:")
x_sv, x_b = jnp.zeros(4).at[0].set(1.0), jnp.zeros(2, jnp.int32)
th0 = [jnp.array([0.0])] * 3
ex0 = np.concatenate([np.ravel(np.array(a)) for a in jax.grad(
    lambda z: sv.expval_all(sv.build_circuit(c2, z), x_sv)[1])(th0)])
for m in METHODS:
    sds = []
    for N in (1000, 4000):
        b = psc.BranchingSimulator(num_samples=N, diff_method=m)
        g = jax.jit(jax.grad(lambda z, k: b.expval_all(b.build_circuit(c2, z), x_b, k)[1]))
        gs = np.array([np.concatenate([np.ravel(np.array(a)) for a in g(th0, jax.random.key(i))]) for i in range(60)])
        sds.append(gs.std(0).mean())
    print(f"  {m:20s} sd(N=1000)={sds[0]:.4f} sd(N=4000)={sds[1]:.4f} ratio={sds[0] / sds[1]:.2f}")

print("\nparam_shift_filter bias (mean - exact, max over 3 thetas), 400 repeats:")
for N in (20, 100, 1000):
    b = psc.BranchingSimulator(num_samples=N, diff_method="param_shift_filter")
    g = jax.jit(jax.grad(lambda z, k: b.expval_all(b.build_circuit(c2, z), x_b, k)[1]))
    gs = np.array([np.concatenate([np.ravel(np.array(a)) for a in g(th0, jax.random.key(i))]) for i in range(400)])
    se = (gs.std(0, ddof=1) / np.sqrt(400)).max()
    print(f"  N={N:>5}: max|bias| = {np.abs(gs.mean(0) - ex0).max():.4f}   (SE of the mean {se:.4f})")

# ---- 3. silent zeros through sample() --------------------------------------------------------
b = psc.BranchingSimulator(num_samples=64)
def via_sample(z, k):
    comp = b.build_circuit(c2, z)
    return b.sample(comp, x_b, k).astype(jnp.float32).mean(0)[1]
print("\njax.grad through sample():", [float(a[0]) for a in jax.grad(via_sample)(th0, jax.random.key(0))],
      " <- exact zeros, no error")

# ---- 4. per-gate REINFORCE (Eq. 34): grad = E[(f - baseline) * sum_j dlogG_j/dtheta_j] -------
th = jnp.array([0.3, -0.4, 0.8])

def run_chain(theta, key):
    """Plain-JAX trajectory of the 3-gate circuit; returns f = final bit 1 and the summed log-probability of branches."""
    k = jax.random.split(key, 3)
    p = jax.nn.sigmoid(theta)
    b = jax.vmap(lambda kk, pp: jax.random.bernoulli(kk, pp))(k, p).astype(jnp.int32)
    z0 = 0
    z0 = z0 ^ b[0]                                   # PNOT(0)
    z1 = 0 ^ (z0 & b[1])                             # PCNOT([0,1]): flip site 1 when site 0 = 1
    z1 = z1 ^ b[2]                                   # PNOT(1)
    logp = jnp.sum(jnp.where(b == 1, jnp.log(p), jnp.log1p(-p)))
    return z1.astype(jnp.float32), logp

def surrogate(theta, keys):
    f, logp = jax.vmap(lambda kk: run_chain(theta, kk))(keys)
    baseline = jax.lax.stop_gradient(f.mean())
    return jnp.mean(jax.lax.stop_gradient(f - baseline) * logp)     # grad of this = REINFORCE with baseline

exact = np.concatenate([np.ravel(np.array(a)) for a in jax.grad(
    lambda z: sv.expval_all(sv.build_circuit(c2, [z[0:1], z[1:2], z[2:3]]), x_sv)[1])(th)])
gfun = jax.jit(jax.grad(surrogate))
for M in (1000, 10000, 100000):
    est = np.array([np.array(gfun(th, jax.random.split(jax.random.key(1000 + i), M))) for i in range(30)])
    print(f"REINFORCE M={M:>6}: mean {est.mean(0).round(4)}  sd {est.std(0).round(4)}   exact {exact.round(4)}")
