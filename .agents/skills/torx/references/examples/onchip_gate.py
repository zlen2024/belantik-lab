"""Try to compute the per-dimension product on the chip with the 2-local couplers.

Per dimension i: three spins (a_i, b_i, c_i), a/b carry q_i/k_i via bias, c_i is an AND
gate built from the Ising penalty lam * P(x,y,z), x,y,z in {0,1} bits:
    P = xy - 2(x+y)z + 3z   (zero iff z = x AND y)
In spins: P = 3/4 - a/4 - b/4 + c/2 + ab/4 - ac/2 - bc/2
Each spin has degree <= 2, so it fits the degree-16 hardwired graph (paper Fig. 7).

With bits independent and means m_a, m_b:  E[c01] = (1 + m_a + m_b + m_a m_b)/4,
so  m_a m_b = 4 E[c01] - 1 - m_a - m_b, and the host only reads the c spins.

Run:  python onchip_gate.py
"""
import jax
import jax.numpy as jnp
import numpy as np
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init


def run_gate(key, ma, mb, lam, n_samples, n_warmup=50):
    d = len(ma)
    A = [SpinNode() for _ in range(d)]
    B = [SpinNode() for _ in range(d)]
    Cn = [SpinNode() for _ in range(d)]
    nodes = A + B + Cn
    edges = [(A[i], B[i]) for i in range(d)] + [(A[i], Cn[i]) for i in range(d)] + \
            [(B[i], Cn[i]) for i in range(d)]
    ha = np.arctanh(ma) + lam / 4
    hb = np.arctanh(mb) + lam / 4
    hc = np.full(d, -lam / 2)
    w = np.concatenate([np.full(d, -lam / 4), np.full(d, lam / 2), np.full(d, lam / 2)])
    model = IsingEBM(nodes, edges, jnp.asarray(np.concatenate([ha, hb, hc]), jnp.float32),
                     jnp.asarray(w, jnp.float32), jnp.array(1.0))
    blocks = [Block(A), Block(B), Block(Cn)]       # triangle -> 3 colours
    program = IsingSamplingProgram(model, blocks, [])
    sched = SamplingSchedule(n_warmup=n_warmup, n_samples=n_samples, steps_per_sample=1)
    k1, k2 = jax.random.split(key)
    init = hinton_init(k1, model, blocks, ())
    out = sample_states(k2, program, sched, init, [], blocks)
    s = [2.0 * np.asarray(o, np.float32) - 1.0 for o in out]
    return s   # [A, B, C] each (n_samples, d)


def estimate(s, ma, mb):
    a, b, c = s
    cb = (1 + c) / 2
    per_sample = np.sum(4 * cb - 1 - ma - mb, axis=1)   # sum_i m_a m_b, one value per sample
    return per_sample.mean(), per_sample, a.mean(0), b.mean(0), cb.mean(0)


if __name__ == "__main__":
    d = 288
    rng = np.random.default_rng(0)
    q = rng.normal(size=d)
    k = 0.5 * q + rng.normal(size=d)
    sq, sk = abs(q).max() * 1.05, abs(k).max() * 1.05
    ma, mb = q / sq, k / sk
    exact = float(np.sum(ma * mb))
    print(f"exact sum m_a m_b = {exact:.2f}  (q.k = {exact * sq * sk:.2f})")
    key = jax.random.key(1)
    N, R = 1000, 6
    print(f"{'lam':>5} {'RMS err':>9} {'bias err':>9} {'marginal shift':>15} {'AND fidelity':>13}")
    for lam in (0.5, 1, 2, 3, 4, 6, 8):
        ests, shifts, fid = [], [], []
        for r in range(R):
            key, kk = jax.random.split(key)
            s = run_gate(kk, ma, mb, lam, N)
            e, ps, ea, eb, ec = estimate(s, ma, mb)
            ests.append(e)
            shifts.append(np.mean(np.abs(np.concatenate([ea - ma, eb - mb]))))
            a, b, c = s
            fid.append(np.mean(((a > 0) & (b > 0)) == (c > 0)))
        ests = np.array(ests)
        print(f"{lam:>5} {np.sqrt(np.mean((ests - exact) ** 2)):>9.2f} {abs(ests.mean() - exact):>9.2f} "
              f"{np.mean(shifts):>15.4f} {np.mean(fid):>13.3f}")
