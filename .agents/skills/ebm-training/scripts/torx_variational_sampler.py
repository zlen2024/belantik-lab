"""Torx circuit as the amortised sampler q_phi of the adversarial / variational EBM objective (Song & Kingma Sec. 6.3).

    max_theta min_phi  E_q[E_theta(x)] - E_data[E_theta(x)] - H(q_phi)            (Eq. 32)

needs a q_phi with (i) fast sampling and (ii) tractable entropy.  A Torx DiscretePCircuit gives both for small n:
`StateVectorSimulator.density` returns the exact q(x) (differentiable in phi), on hardware the same circuit samples.

What this script verifies (n = 5 spins, everything exact, so any gap is model bias, not Monte Carlo noise):
  1. inner step: min_phi KL(q_phi || p_theta) = E_q[E] - H(q) + log Z.  Independent-bit q (PNOT only) vs a q with
     PCNOT between every pair.  The residual KL measures expressivity.
     (the same print also reports the outer-step gradient bias ||E_q[grad E] - E_p[grad E]|| of using q instead of p_theta)
  2. alternating max-min loop vs exact MLE (negative phase = exact p_theta): final KL(p_teacher || p_theta).

Run from OUTSIDE master/ (the torx/ repo folder shadows the package):  cd /tmp && python <this file>
Runtime about 1-2 min on CPU.
"""
import argparse
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax
from torx import psc

N = 5
STATES = np.array([[(i >> (N - 1 - k)) & 1 for k in range(N)] for i in range(2**N)])  # site 0 = MSB
SPIN = jnp.array(2 * STATES - 1, dtype=jnp.float32)
PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
PAIR_FEAT = jnp.stack([SPIN[:, i] * SPIN[:, j] for i, j in PAIRS], 1)  # (2^N, n_pairs)
X0 = jnp.zeros(2**N).at[0].set(1.0)  # all bits 0


def energy(h, Jv):  # E(s) = -(h.s + sum_{i<j} J_ij s_i s_j)
    return -(SPIN @ h + PAIR_FEAT @ Jv)


def make_circuit(kind):
    gates = [psc.PNOT(i) for i in range(N)]
    if kind == "pairs":
        gates += [psc.PCNOT([i, j]) for i, j in PAIRS]
    return psc.DiscretePCircuit(gates), len(gates)


SV = psc.StateVectorSimulator()


def make_q(kind):
    circ, G = make_circuit(kind)

    def q_of(phi):
        return SV.density(SV.build_circuit(circ, [phi[g : g + 1] for g in range(G)]), X0)

    return q_of, G


def kl_q_p(q, E):
    logp = -E - jax.scipy.special.logsumexp(-E)
    return jnp.sum(q * (jnp.log(jnp.clip(q, 1e-12)) - logp))


def fit_q(q_of, phi, E, steps, lr=0.05):
    opt = optax.adam(lr)
    ost = opt.init(phi)
    loss = lambda p: kl_q_p(q_of(p), E)  # = E_q[E] - H(q) + log Z
    @jax.jit
    def step(p, o):
        l, g = jax.value_and_grad(loss)(p)
        u, o = opt.update(g, o, p)
        return optax.apply_updates(p, u), o, l

    for _ in range(steps):
        phi, ost, l = step(phi, ost)
    return phi, float(l)


def teacher(seed=0):
    rng = np.random.default_rng(seed)
    h = jnp.array(0.4 * rng.standard_normal(N), dtype=jnp.float32)
    Jv = jnp.array(0.9 * rng.standard_normal(len(PAIRS)), dtype=jnp.float32)
    p = jax.nn.softmax(-energy(h, Jv))
    return h, Jv, p


def grad_E_moments(w):  # E_w[grad_theta E] for weights w over states: (-<s_i>, -<s_i s_j>)
    return -(w @ SPIN), -(w @ PAIR_FEAT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outer", type=int, default=600)
    ap.add_argument("--inner", type=int, default=30, help="phi steps per theta step; too few -> q lags -> divergence")
    ap.add_argument("--lr_theta", type=float, default=0.01, help="0.05 diverged in testing (KL 7-21)")
    a = ap.parse_args()

    h_t, J_t, p_t = teacher(a.seed)
    print(f"n={N} spins, teacher entropy {float(-(p_t*jnp.log(p_t)).sum()):.3f} nats (max {N*np.log(2):.3f})")

    # ---------- 1. inner step: how well can each q represent a fixed Boltzmann target? ----------
    key = jax.random.key(a.seed)
    print("\n[1] min_phi KL(q_phi || p_teacher)")
    for kind in ["indep", "pairs"]:
        q_of, G = make_q(kind)
        phi0 = 0.1 * jax.random.normal(key, (G,))
        phi, kl = fit_q(q_of, phi0, energy(h_t, J_t), 1500)
        q = q_of(phi)
        bias_b, bias_J = (x - y for x, y in zip(grad_E_moments(q), grad_E_moments(p_t)))
        bias = float(jnp.linalg.norm(jnp.concatenate([bias_b, bias_J])))
        print(f"  {kind:5s} circuit: {G:2d} gates  KL(q||p) = {kl:.4f}   ||E_q[gradE]-E_p[gradE]|| = {bias:.4f}")

    # ---------- 3. alternating max-min vs exact MLE; q warm-started, --inner phi steps per theta step ----------
    print("\n[2] learn a student from teacher data (infinite data: positive phase exact)")
    pos_b, pos_J = grad_E_moments(p_t)  # E_data[grad E]
    results = {}
    for name in ["exact_mle", "q_indep", "q_pairs"]:
        h = jnp.zeros(N)
        Jv = jnp.zeros(len(PAIRS))
        opt = optax.adam(a.lr_theta)
        ost = opt.init((h, Jv))
        if name != "exact_mle":
            q_of, G = make_q("indep" if name == "q_indep" else "pairs")
            phi = 0.1 * jax.random.normal(key, (G,))
            opt_q = optax.adam(0.05)
            ost_q = opt_q.init(phi)

            @jax.jit
            def outer(h, Jv, phi, ost, ost_q, q_of=q_of, opt_q=opt_q):
                E = energy(h, Jv)
                for _ in range(a.inner):
                    g = jax.grad(lambda p: kl_q_p(q_of(p), E))(phi)
                    u, ost_q = opt_q.update(g, ost_q, phi)
                    phi = optax.apply_updates(phi, u)
                neg_b, neg_J = grad_E_moments(q_of(phi))
                gh, gJ = pos_b - neg_b, pos_J - neg_J  # grad of -loglik: E_data[gradE] - E_neg[gradE]
                u, ost = opt.update((gh, gJ), ost, (h, Jv))
                h, Jv = optax.apply_updates((h, Jv), u)
                return h, Jv, phi, ost, ost_q

        else:

            @jax.jit
            def outer_exact(h, Jv, ost):
                neg_b, neg_J = grad_E_moments(jax.nn.softmax(-energy(h, Jv)))
                u, ost = opt.update((pos_b - neg_b, pos_J - neg_J), ost, (h, Jv))
                h, Jv = optax.apply_updates((h, Jv), u)
                return h, Jv, ost

        for t in range(a.outer):
            if name == "exact_mle":
                h, Jv, ost = outer_exact(h, Jv, ost)
            else:
                h, Jv, phi, ost, ost_q = outer(h, Jv, phi, ost, ost_q)
        p_s = jax.nn.softmax(-energy(h, Jv))
        kl = float(jnp.sum(p_t * (jnp.log(p_t) - jnp.log(p_s))))
        extra = ""
        if name != "exact_mle":
            extra = f"   final KL(q||p_theta) = {float(kl_q_p(q_of(phi), energy(h, Jv))):.4f}"
        results[name] = kl
        print(f"  {name:9s} KL(p_teacher || p_student) = {kl:.4f}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
