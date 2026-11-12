"""Cookbook 8: THRML-emulated attention with DAC quantisation, a one-hot softmax prototype, correlated samples.

Uses THRML (not Torx) because Z1 is an Ising Gibbs sampler. Run from OUTSIDE master/.
Needs z1_attention.py in the same folder (imports its p-bit sampler).
"""
import itertools, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax, jax.numpy as jnp, numpy as np
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from z1_attention import Z1Config, attention_B, softmax

# ---- A. THRML attention with a bias DAC of n bits (E[s] = tanh(theta), |theta| <= 3) -----------------
rng = np.random.default_rng(0)
d, Kk = 288, 8
q = rng.normal(size=d).astype(np.float32)
Km = (rng.normal(size=(Kk, d)) * 0.3).astype(np.float32)
qh = q / np.linalg.norm(q)
target = np.linspace(-1.5, 1.5, Kk) * np.sqrt(d)
for j in range(Kk):
    Km[j] += ((target[j] - Km[j] @ q) / np.linalg.norm(q)) * qh
true_w = softmax(Km @ q / np.sqrt(d))
print(f"A. THRML attention, N=10000, 6 repeats, mean TV +/- 95% t-CI (t=2.571); TV is between softmax WEIGHTS, not a histogram, so no multinomial floor applies; true max weight {true_w.max():.2f}")
key = jax.random.key(0)
for bits in (4, 6, 8, 16):
    tvs = []
    for _ in range(6):
        key, ks = jax.random.split(key)
        _, sc = attention_B(ks, Z1Config(bias_bits=bits), q, Km, 10000)
        tvs.append(0.5 * np.abs(softmax(sc) - true_w).sum())
    print(f"   bias DAC {bits:>2} bits: TV = {np.mean(tvs):.4f} +/- {2.571 * np.std(tvs, ddof=1) / np.sqrt(6):.4f}")

# ---- B. one-hot softmax: sample a key index with K spins and all-to-all couplers --------------------------
# x_k in {0,1}: E(x) = -sum theta_k x_k + lam (sum x - 1)^2   ->  one-hot states have probability  softmax(theta) EXACTLY.
# Spins s = 2x - 1 in THRML convention E = -(sum b s + sum J s s):  b_k = theta_k/2 + lam (2-K)/2,  J = -lam/2.
K = 8
theta = np.linspace(-1.5, 1.5, K)
w_true = softmax(theta)


def ising_params(lam):
    return theta / 2 + lam * (2 - K) / 2, -lam / 2


def exact_dist(lam):
    b, J = ising_params(lam)
    S = np.array(list(itertools.product([-1, 1], repeat=K)))
    E = -(S @ b + J * (np.sum(S, 1) ** 2 - K) / 2)            # sum_{k<l} s_k s_l = ((sum s)^2 - K)/2
    P = np.exp(-E); P /= P.sum()
    onehot = (S == 1).sum(1) == 1
    return S, P, onehot

print("\nB. one-hot softmax prototype, K=8 (degree needed: K-1 = 7 <= 16)")
print(f"{'lam':>5} {'mass on one-hot':>16} {'TV(conditional, softmax)':>26}")
for lam in (1, 2, 4, 6):
    S, P, oh = exact_dist(lam)
    cond = P[oh] / P[oh].sum()
    idx = np.argmax(S[oh] == 1, axis=1)                         # which spin is up
    cond_by_k = np.zeros(K); cond_by_k[idx] = cond
    print(f"{lam:>5} {P[oh].sum():>16.3f} {0.5 * np.abs(cond_by_k - w_true).sum():>26.2e}")

# Gibbs sampling in THRML: leakage and mixing
print("\n   THRML block Gibbs, one block per spin (complete graph needs K colours), 20000 samples:")
for lam in (2, 4, 6):
    b, J = ising_params(lam)
    nodes = [SpinNode() for _ in range(K)]
    edges = [(nodes[i], nodes[j]) for i in range(K) for j in range(i + 1, K)]
    model = IsingEBM(nodes, edges, jnp.asarray(b, jnp.float32), jnp.full((len(edges),), J, jnp.float32), jnp.array(1.0))
    blocks = [Block([n]) for n in nodes]
    prog = IsingSamplingProgram(model, blocks, [])
    sched = SamplingSchedule(n_warmup=500, n_samples=20000, steps_per_sample=1)
    k1, k2 = jax.random.split(jax.random.key(lam))
    init = hinton_init(k1, model, blocks, ())
    out = sample_states(k2, prog, sched, init, [], [Block(nodes)])[0]
    s = 2.0 * np.asarray(out, np.float32) - 1.0
    oh = (s == 1).sum(1) == 1
    freq = np.bincount(np.argmax(s[oh] == 1, axis=1), minlength=K) / max(oh.sum(), 1)
    stay = np.mean(np.all(s[1:] == s[:-1], axis=1))
    print(f"   lam={lam}: one-hot fraction {oh.mean():.3f}, TV(empirical | one-hot, softmax) {0.5 * np.abs(freq - w_true).sum():.3f}, "
          f"consecutive samples identical {stay:.2f}")

# ---- C. correlated samples: effective sample size ---------------------------------------------------------
print("\nC. correlated samples (lag-1 correlation rho, e.g. rho = exp(-dt/tau0) for an X0-like pbit, tau0 ~ 100 ns)")
m, N, R = 0.3, 2000, 400
r = np.random.default_rng(1)
for rho in (0.0, 0.5, 0.9):
    means = []
    for _ in range(R):
        keep = r.random(N) < rho                                  # stay frozen with prob rho else redraw: lag-k corr = rho^k
        fresh = np.where(r.random(N) < (1 + m) / 2, 1.0, -1.0)
        x = np.empty(N); x[0] = fresh[0]
        for t in range(1, N):
            x[t] = x[t - 1] if keep[t] else fresh[t]
        means.append(x.mean())
    var_iid = (1 - m ** 2) / N
    print(f"   rho={rho}: var(mean) measured {np.var(means):.2e}, iid formula x (1+rho)/(1-rho) = {var_iid * (1 + rho) / (1 - rho):.2e}, "
          f"N_eff = N (1-rho)/(1+rho) = {N * (1 - rho) / (1 + rho):.0f}")
for dt in (300, 100, 30, 10):
    rho = np.exp(-dt / 100)
    print(f"   sampling every {dt:>3} ns with tau0=100 ns: rho={rho:.2f}, N_eff/N = {(1 - rho) / (1 + rho):.2f}")
