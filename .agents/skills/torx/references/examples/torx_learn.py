"""Learning torx step by step, aimed at the p-bit dot-product idea.

Run from OUTSIDE the master/ folder (the torx/ repo folder there shadows the package):
    cd /tmp && python <path>/attn_thrml/torx_learn.py

1. PNOT as a p-bit           : P(bit=1) = sigmoid(theta)  ->  spin mean m = tanh(theta/2)
2. Gates are truth tables    : PCNOT / POR with a near-deterministic theta
3. PISING pair               : coupling J moves the marginals (why 2-local product fails)
4. Sampled dot product       : 2d PNOT p-bits via BranchingSimulator, error ~ 1/sqrt(N)
"""
import jax
import jax.numpy as jnp
import numpy as np
from torx import psc

BIG = 20.0   # theta -> +inf makes the gate deterministic (p = sigmoid(20) ~ 1)


def section(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------- 1. PNOT as a p-bit
section("1. PNOT as a p-bit (start in 0, flip with p = sigmoid(theta))")
sim = psc.StateVectorSimulator()
for m in (-0.8, 0.0, 0.5):
    theta = 2 * np.arctanh(m)                       # m = tanh(theta/2)
    circ = psc.DiscretePCircuit([psc.PNOT(0)])
    comp = sim.build_circuit(circ, [jnp.array([theta])])
    d = np.array(sim.density(comp, jnp.array([1.0, 0.0])))   # [P(0), P(1)]
    print(f"target spin mean {m:+.2f}  theta={theta:+.3f}  P(1)={d[1]:.4f}  "
          f"spin mean={2 * d[1] - 1:+.4f}")

# ---------------------------------------------------------------- 2. gates = truth tables
section("2. PCNOT and POR as truth tables (theta = +20, state index = bit string)")
for name, gate in (("PCNOT", psc.PCNOT([0, 1])), ("POR", psc.POR([0, 1]))):
    circ = psc.DiscretePCircuit([gate])
    comp = sim.build_circuit(circ, [jnp.array([BIG])])
    print(name)
    for i, label in enumerate(("00", "01", "10", "11")):
        x = jnp.zeros(4).at[i].set(1.0)
        d = np.array(sim.density(comp, x))
        print(f"  {label} -> {('00', '01', '10', '11')[int(d.argmax())]}   (p={d.max():.3f})")
print("No AND gate here: POR sets bit0 = OR(bit0, bit1) and clears bit1.")

# ---------------------------------------------------------------- 3. PISING pair
section("3. PISING pair: does coupling J make <ab> = m_a * m_b ?  (beta=1, dt=50)")
ma, mb = 0.6, -0.4
for J in (0.0, 0.3, 1.0):
    circ = psc.DiscretePCircuit([psc.PISING([0, 1])])
    comp = sim.build_circuit(circ, [jnp.array([J, np.arctanh(ma), np.arctanh(mb), 1.0, 50.0])])
    d = np.array(sim.density(comp, jnp.array([1.0, 0, 0, 0])))
    s = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    print(f"J={J:3.1f}  <ab>={float(d @ (s[:, 0] * s[:, 1])):+.4f}  target {ma * mb:+.4f}  "
          f"<a>={float(d @ s[:, 0]):+.3f} (want {ma:+.1f})  <b>={float(d @ s[:, 1]):+.3f} (want {mb:+.1f})")

# ---------------------------------------------------------------- 4. sampled dot product
section("4. Sampled dot product with PNOT p-bits (BranchingSimulator)")
d_dim = 288
rng = np.random.default_rng(0)
q = rng.normal(size=d_dim)
k = 0.5 * q + rng.normal(size=d_dim)
sq, sk = abs(q).max() * 1.05, abs(k).max() * 1.05
ma, mb = q / sq, k / sk
exact = float(q @ k)
thetas = [jnp.array([2 * np.arctanh(m)]) for m in np.concatenate([ma, mb])]
circ = psc.DiscretePCircuit([psc.PNOT(i) for i in range(2 * d_dim)])
print(f"d={d_dim}, {2 * d_dim} p-bits, exact q.k = {exact:.2f}")
key = jax.random.key(0)
x0 = jnp.zeros(2 * d_dim, dtype=jnp.int32)
for N in (10, 100, 1000, 10000):
    errs = []
    bsim = psc.BranchingSimulator(num_samples=N)
    comp = bsim.build_circuit(circ, thetas)
    for _ in range(6):
        key, kk = jax.random.split(key)
        s = 2.0 * np.asarray(bsim.sample(comp, x0, kk), dtype=np.float32) - 1.0
        est = float(np.mean(np.sum(s[:, :d_dim] * s[:, d_dim:], axis=1))) * sq * sk
        errs.append(est - exact)
    print(f"N={N:>6}  RMS error = {np.sqrt(np.mean(np.square(errs))):6.2f}")
