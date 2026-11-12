"""Dot-product attention on an emulated Extropic Z1 (THRML as the p-bit simulator).

Z1 is emulated by constraints, not by hardware: binary spins, no couplings used,
quantized bias DAC, fixed beta, and a p-bit budget. Numbers in Z1Config marked
ASSUMED are placeholders -- replace them with the real datasheet values.

Two estimators of q.k (d = 288):
  A  shared coins : d fair-coin p-bits z_i (bias 0). u=sum q_i z_i, v=sum k_i z_i,
                    E[u v] = q.k because E[z_i z_j] = delta_ij.
  B  stochastic   : 2d independent p-bits, E[s_i] = q_i (resp. k_i) set by the bias.
                    E[sum_i sq_i sk_i] = q.k.

Run:  python z1_attention.py
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init


@dataclass(frozen=True)
class Z1Config:
    degree: int = 16         # arXiv 2608.01612 Fig. 7: hardwired degree-16 sparse graph, 2-local couplings
    n_pbits: int = 269_568   # extropic.ai "From One to One Billion": Z1 die, 8 cores, <1 W, >50 MHz
    bias_bits: int = 8       # ASSUMED bias DAC resolution
    bias_max: float = 3.0    # ASSUMED |theta| range of the DAC (tanh(3)=0.995)
    beta: float = 1.0        # fixed on chip


def quantize_bias(theta, cfg):
    levels = 2 ** cfg.bias_bits - 1
    t = np.clip(theta, -cfg.bias_max, cfg.bias_max)
    step = 2 * cfg.bias_max / levels
    return np.round(t / step) * step


def run_pbits(key, cfg, biases, n_samples, n_warmup=5):
    """Independent p-bits (no edges). Returns spins in {-1,+1}, shape (n_samples, n)."""
    n = len(biases)
    assert n <= cfg.n_pbits, f"needs {n} p-bits, chip has {cfg.n_pbits}"
    nodes = [SpinNode() for _ in range(n)]
    # IsingEBM rejects an empty edge list, so keep one weight-0 edge (node0-node1) and
    # put its endpoints in different blocks. The p-bits stay fully independent.
    blocks = [Block(nodes[:1] + nodes[2:]), Block(nodes[1:2])]
    model = IsingEBM(nodes, [(nodes[0], nodes[1])], jnp.asarray(biases, jnp.float32),
                     jnp.zeros((1,), jnp.float32), jnp.array(cfg.beta))
    program = IsingSamplingProgram(model, blocks, [])
    schedule = SamplingSchedule(n_warmup=n_warmup, n_samples=n_samples, steps_per_sample=1)
    k_init, k_run = jax.random.split(key)
    init = hinton_init(k_init, model, blocks, ())
    out = sample_states(k_run, program, schedule, init, [], [Block(nodes)])[0]
    return 2.0 * np.asarray(out, dtype=np.float32) - 1.0


def theta_for_mean(m, cfg):
    """Bias giving E[s]=m under P(s=+1)=sigmoid(2*beta*theta)."""
    m = np.clip(m, -0.999, 0.999)
    return quantize_bias(np.arctanh(m) / cfg.beta, cfg)


def dot_A(key, cfg, q, k, n_samples):
    z = run_pbits(key, cfg, np.zeros(len(q)), n_samples)
    return float(np.mean((z @ q) * (z @ k)))


def dot_B(key, cfg, q, k, n_samples):
    sq, sk = abs(q).max() * 1.05, abs(k).max() * 1.05   # map into (-1,1)
    biases = np.concatenate([theta_for_mean(q / sq, cfg), theta_for_mean(k / sk, cfg)])
    s = run_pbits(key, cfg, biases, n_samples)
    d = len(q)
    return float(np.mean(np.sum(s[:, :d] * s[:, d:], axis=1))) * sq * sk


def attention_B(key, cfg, q, K, n_samples):
    """softmax(q.K^T / sqrt(d)) with one shared q block + one p-bit block per key."""
    d = len(q)
    sq = abs(q).max() * 1.05
    sk = abs(K).max() * 1.05
    biases = np.concatenate([theta_for_mean(q / sq, cfg)] +
                            [theta_for_mean(kj / sk, cfg) for kj in K])
    s = run_pbits(key, cfg, biases, n_samples)
    sq_bits = s[:, :d]
    scores = np.array([np.mean(np.sum(sq_bits * s[:, d * (j + 1):d * (j + 2)], axis=1))
                       for j in range(len(K))]) * sq * sk
    return scores, scores / np.sqrt(d)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


if __name__ == "__main__":
    cfg = Z1Config()
    d = 288
    rng = np.random.default_rng(0)
    q = rng.normal(size=d).astype(np.float32)
    k = (0.5 * q + rng.normal(size=d)).astype(np.float32)
    exact = float(q @ k)
    print(f"d={d}  exact q.k = {exact:.2f}  (chip budget {cfg.n_pbits} p-bits)\n")

    key = jax.random.key(0)
    print(f"{'N':>6} {'A est':>10} {'A |err|':>9} {'B est':>10} {'B |err|':>9}")
    for N in (10, 100, 1000, 10000):
        errA, errB = [], []
        for r in range(8):   # repeat to see the RMS error, which should follow 1/sqrt(N)
            key, ka, kb = jax.random.split(key, 3)
            errA.append(dot_A(ka, cfg, q, k, N) - exact)
            errB.append(dot_B(kb, cfg, q, k, N) - exact)
        print(f"{N:>6} {exact + np.mean(errA):>10.2f} {np.sqrt(np.mean(np.square(errA))):>9.2f} "
              f"{exact + np.mean(errB):>10.2f} {np.sqrt(np.mean(np.square(errB))):>9.2f}")

    # Attention: 1 query vs 8 keys, method B on a single emulated chip run.
    # Keys are built so the logits q.k_j/sqrt(d) are spread over [-1.5, 1.5] (max weight about 0.36);
    # a saturated softmax (one weight ~ 1) would make |dw| = 0 at every N and test nothing.
    K = (rng.normal(size=(8, d)) * 0.3).astype(np.float32)
    qh = q / np.linalg.norm(q)
    target = np.linspace(-1.5, 1.5, 8) * np.sqrt(d)               # q.k_j targets (logit * sqrt(d))
    for j in range(8):
        K[j] += ((target[j] - K[j] @ q) / np.linalg.norm(q)) * qh
    true_w = softmax((K @ q) / np.sqrt(d))
    assert true_w.max() < 0.99, "saturated softmax: this test would be vacuous"
    print(f"\nattention weights (true max {true_w.max():.2f}); TV(true, Z1-emulated B), 8 repeats, mean")
    for N in (100, 1000, 10000):
        tvs = []
        for _ in range(8):
            key, ks = jax.random.split(key)
            _, sc = attention_B(ks, cfg, q, K, N)
            tvs.append(0.5 * np.abs(softmax(sc) - true_w).sum())
        print(f"N={N:>5}  TV = {np.mean(tvs):.4f}  p-bits used = {d * 9}")
