"""Forward AIS and reverse AIS (RAISE-style) estimates of log Z for a spin RBM, checked against the exact value.

Why: beyond ~24 spins there is no enumeration, and the usual advice is "use AIS". This script shows what AIS actually
delivers and how to bracket it (Neal 2001; Salakhutdinov & Murray 2008; Burda, Grosse, Salakhutdinov 2015):

  * forward AIS: log Z_hat = log mean_i exp(log w_i) + log Z_0.  E[Z_hat] = Z, so by Jensen log Z_hat is a stochastic
    LOWER bound on log Z (held-out log-likelihood computed from it is optimistic).
  * reverse AIS started from EXACT samples of the target (available here because the model is small enough to
    enumerate; for large models use RAISE, which starts from an approximate posterior sample): log Z_hat_rev =
    log Z_0 - log mean exp(log w_rev), a stochastic UPPER bound.
  * the exact log Z (enumeration of all 2^14 states) must lie inside the bracket, up to Monte Carlo spread.

Path: geometric from the uniform distribution (beta=0, log Z_0 = n log 2) to the RBM (beta=1), f_beta = exp(beta (b.s + sum J s s)).
Transitions: exact block Gibbs for the spin RBM (vectorised, pure JAX).  Coupling scales 1.5 and 7 are easy cases; a large scale makes the landscape rugged and shows what an untrustworthy bracket looks like.

Run from OUTSIDE master/:  cd /tmp && python -I <this file>      (about 1-2 min on CPU)
"""
import itertools

import jax
import jax.numpy as jnp
import numpy as np

N_VIS, N_LAT = 8, 6


def make_rbm(seed, strength):
    rng = np.random.default_rng(seed)
    J = strength * rng.choice([-1.0, 1.0], size=(N_VIS, N_LAT)) / np.sqrt(N_VIS)
    bv = 0.05 * rng.standard_normal(N_VIS)
    bh = 0.05 * rng.standard_normal(N_LAT)
    return jnp.array(bv, jnp.float32), jnp.array(bh, jnp.float32), jnp.array(J, jnp.float32)


def neg_energy(bv, bh, J, v, h):
    return v @ bv + h @ bh + jnp.einsum("...i,ij,...j->...", v, J, h)


def exact_logZ_and_sampler(bv, bh, J):
    S = np.array(list(itertools.product([-1.0, 1.0], repeat=N_VIS + N_LAT)), dtype=np.float32)
    ne = neg_energy(bv, bh, J, jnp.array(S[:, :N_VIS]), jnp.array(S[:, N_VIS:]))
    logZ = float(jax.scipy.special.logsumexp(ne))
    return logZ, S, np.asarray(jax.nn.softmax(ne))


def gibbs(key, v, h, beta, bv, bh, J):
    k1, k2 = jax.random.split(key)
    h = 2.0 * jax.random.bernoulli(k1, jax.nn.sigmoid(2 * beta * (bh + v @ J))).astype(jnp.float32) - 1.0
    v = 2.0 * jax.random.bernoulli(k2, jax.nn.sigmoid(2 * beta * (bv + h @ J.T))).astype(jnp.float32) - 1.0
    return v, h


def ais(key, v, h, betas, bv, bh, J, reverse):
    """Return log-weights per chain. Forward: betas increasing 0->1 starting from uniform. Reverse: start from exact
    samples at beta=1 and anneal down, weights f_{k-1}/f_k at x_{k-1}."""
    K = len(betas) - 1
    logw = jnp.zeros(v.shape[0])
    ks = jax.random.split(key, K)

    def fwd(c, inp):
        v, h, lw = c
        k, b_prev, b_new = inp
        lw = lw + (b_new - b_prev) * neg_energy(bv, bh, J, v, h)
        v, h = gibbs(k, v, h, b_new, bv, bh, J)
        return (v, h, lw), None

    def rev(c, inp):
        v, h, lw = c
        k, b_hi, b_lo = inp  # step from beta_k=b_hi down to beta_{k-1}=b_lo
        v, h = gibbs(k, v, h, b_hi, bv, bh, J)
        lw = lw + (b_lo - b_hi) * neg_energy(bv, bh, J, v, h)
        return (v, h, lw), None

    b = jnp.array(betas, jnp.float32)
    if not reverse:
        (v, h, logw), _ = jax.lax.scan(fwd, (v, h, logw), (ks, b[:-1], b[1:]))
    else:
        (v, h, logw), _ = jax.lax.scan(rev, (v, h, logw), (ks[::-1], b[1:][::-1], b[:-1][::-1]))
    return logw


def lme(x):
    return float(jax.scipy.special.logsumexp(x) - jnp.log(x.shape[0]))


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="1.5,7.0,25.0", help="coupling scales; large = rugged landscape")
    ap.add_argument("--ks", default="10,100,1000,5000")
    ap.add_argument("--chains", type=int, default=256)
    ap.add_argument("--reps", type=int, default=8)
    a = ap.parse_args()
    M, REPS = a.chains, a.reps
    for strength in [float(v) for v in a.scales.split(",")]:
        bv, bh, J = make_rbm(0, strength)
        logZ, S, p = exact_logZ_and_sampler(bv, bh, J)
        logZ0 = (N_VIS + N_LAT) * np.log(2.0)
        print(f"\ncoupling scale {strength}: exact log Z = {logZ:.3f}  (log Z_0 = {logZ0:.3f})")
        print(f"{'K temps':>8s} {'forward (lower)':>20s} {'reverse (upper)':>20s}   exact within bracket (3 s.e.)?")
        for K in [int(v) for v in a.ks.split(",")]:
            betas = np.linspace(0.0, 1.0, K + 1)
            fw, rv = [], []
            for r in range(REPS):
                key = jax.random.key(1000 * K + r)
                k1, k2, k3 = jax.random.split(key, 3)
                v0 = 2.0 * jax.random.bernoulli(k1, 0.5, (M, N_VIS)).astype(jnp.float32) - 1.0
                h0 = 2.0 * jax.random.bernoulli(k2, 0.5, (M, N_LAT)).astype(jnp.float32) - 1.0
                fw.append(logZ0 + lme(ais(k3, v0, h0, betas, bv, bh, J, reverse=False)))
                idx = np.random.default_rng(r + K).choice(len(p), M, p=p / p.sum())
                vs, hs = jnp.array(S[idx, :N_VIS]), jnp.array(S[idx, N_VIS:])
                lw = ais(jax.random.fold_in(key, 7), vs, hs, betas, bv, bh, J, reverse=True)
                rv.append(logZ0 - lme(lw))
            fw, rv = np.array(fw), np.array(rv)
            se_f, se_r = fw.std() / np.sqrt(REPS), rv.std() / np.sqrt(REPS)
            inside = fw.mean() - 3 * se_f <= logZ <= rv.mean() + 3 * se_r  # consistent with the bracket at 3 standard errors
            print(f"{K:8d} {fw.mean():9.3f} +- {fw.std():5.3f} {rv.mean():11.3f} +- {rv.std():5.3f}   {'yes' if inside else 'NO'}")
    print("\nRead: forward <= exact <= reverse in expectation; the gap closes as K grows; a wide gap at the largest K means the")
    print("estimate is not trustworthy for that model. Report both, with K, chains and the spread over replicates.")


if __name__ == "__main__":
    main()
