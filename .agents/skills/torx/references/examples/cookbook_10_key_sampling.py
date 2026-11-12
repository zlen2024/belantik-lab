"""Cookbook 10: end-to-end key-index sampling with a one-hot Ising sampler (THRML emulation).

Host (digital): logits z_j = q.k_j / sqrt(d)  ->  biases b_k = z_k/2 + lam (2-K)/2, couplers J = -lam/2.
Sampler (emulated Z1): block Gibbs on K spins, all-to-all couplers (degree K-1 = 7).
Readout: index of the single up-spin; leakage (not one-hot) is discarded.
Reports: leakage, TV(empirical, exact softmax) next to the multinomial TV floor for the same number of one-hot
samples, effective sample size from lag-1 autocorrelation, and the effect of an n-bit bias DAC.
Nothing here is hardware; the host still computes the dot products. Run from OUTSIDE master/.
"""
import jax, jax.numpy as jnp, numpy as np
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

rng = np.random.default_rng(0)
d, K, lam = 288, 8, 3.0
q = rng.normal(size=d)
Km = rng.normal(size=(K, d)) * 0.3
qh = q / np.linalg.norm(q)
target = np.linspace(-1.5, 1.5, K) * np.sqrt(d)
for j in range(K):
    Km[j] += ((target[j] - Km[j] @ q) / np.linalg.norm(q)) * qh
z = Km @ q / np.sqrt(d)                                       # host computes the logits
w_true = np.exp(z - z.max()); w_true /= w_true.sum()
print(f"logits {z.round(2)}  true max weight {w_true.max():.2f}")

b_ideal = z / 2 + lam * (2 - K) / 2
J = -lam / 2


def quantise(b, bits, rng_=10.0):
    if bits is None:
        return b
    step = 2 * rng_ / (2 ** bits - 1)
    return np.clip(np.round(b / step) * step, -rng_, rng_)


def run(b, n_samples, steps, seed):
    nodes = [SpinNode() for _ in range(K)]
    edges = [(nodes[i], nodes[j]) for i in range(K) for j in range(i + 1, K)]
    model = IsingEBM(nodes, edges, jnp.asarray(b, jnp.float32), jnp.full((len(edges),), J, jnp.float32), jnp.array(1.0))
    blocks = [Block([n]) for n in nodes]                      # complete graph: one colour per spin
    prog = IsingSamplingProgram(model, blocks, [])
    sched = SamplingSchedule(n_warmup=500, n_samples=n_samples, steps_per_sample=steps)
    k1, k2 = jax.random.split(jax.random.key(seed))
    out = sample_states(k2, prog, sched, hinton_init(k1, model, blocks, ()), [], [Block(nodes)])[0]
    return np.asarray(out)                                    # (n_samples, K) bool, True = +1


def analyse(s):
    up = s.astype(int)
    oh = up.sum(1) == 1
    idx = np.argmax(up[oh], axis=1)                       # index series of the one-hot samples, in time order
    freq = np.bincount(idx, minlength=K) / max(oh.sum(), 1)
    tv = 0.5 * np.abs(freq - w_true).sum()
    r = np.random.default_rng(1)
    floor = np.mean([0.5 * np.abs(r.multinomial(oh.sum(), w_true) / oh.sum() - w_true).sum() for _ in range(200)])
    ind = np.eye(K)[idx]                                  # indicator series of the INDEX (what the histogram uses)
    rho = np.mean([np.corrcoef(ind[1:, k], ind[:-1, k])[0, 1] for k in range(K)])
    n_eff = min(len(idx), len(idx) * (1 - rho) / (1 + rho))
    return 1 - oh.mean(), tv, floor, rho, n_eff


N = 20000
SEEDS = 12
print(f"\nlam={lam}, N={N} recorded samples, ideal biases, mean +/- sd over {SEEDS} seeds (rho and N_eff from the INDEX series)")
print(f"{'steps/sample':>12} {'leakage':>9} {'TV':>15} {'TV floor':>9} {'lag-1 rho':>10} {'N_eff':>8}")
for steps in (1, 5, 20):
    rows = np.array([analyse(run(b_ideal, N, steps, seed=100 * steps + sd)) for sd in range(SEEDS)])
    m, sdv = rows.mean(0), rows.std(0, ddof=1)
    print(f"{steps:>12} {m[0]:>9.3f} {m[1]:>8.4f} +/- {sdv[1]:.4f} {m[2]:>9.4f} {m[3]:>10.3f} {m[4]:>8.0f}")

print(f"\nbias DAC (range +/-10; bias step {20 / 255:.3f} at 8 bits = {2 * 20 / 255:.3f} in logit units), steps/sample=5, mean over {SEEDS} seeds:")
for bits in (4, 6, 8, None):
    rows = np.array([analyse(run(quantise(b_ideal, bits), N, 5, seed=1000 + sd)) for sd in range(SEEDS)])
    print(f"   {str(bits) + '-bit' if bits else 'ideal':>6}: leakage {rows[:, 0].mean():.3f}  TV {rows[:, 1].mean():.4f} +/- {rows[:, 1].std(ddof=1):.4f} (floor {rows[:, 2].mean():.4f})")
