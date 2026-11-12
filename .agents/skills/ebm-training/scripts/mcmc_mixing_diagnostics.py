"""MCMC mixing diagnostics and parallel tempering for a strongly coupled Boltzmann machine (THRML).

Why: every MCMC-trained EBM is only as good as its sampler (Song & Kingma Sec. 3; Nijkamp et al. 2019). Comparing
a few moments is not enough, so this script shows the standard diagnostics on a model whose exact law is known
(8 visible + 6 latent spins, 16384 states by enumeration):

  * split R-hat across independent chains         (>1.05 = chains disagree = not mixed)
  * integrated autocorrelation time and ESS       (Geyer initial positive sequence)
  * total variation of the pooled visible histogram to the EXACT p(v), next to the iid-sample floor
Note: TV of a non-mixing sampler is not monotone in run length (it counts which basins the chains landed in); use several
--key-seed values. Split R-hat and ESS here are not rank-normalised: for heavy-tailed statistics use Vehtari et al. 2021 / ArviZ.
Caution (measured): a WITHIN-chain ESS (each chain centred on its own mean) looks excellent for chains trapped in different
basins while R-hat is large. The multi-chain ESS used here includes between-chain variance and stays tiny. Also report R-hat.
and compares plain block Gibbs at beta=1 with parallel tempering (replica exchange) over a beta ladder.
Swap acceptance for neighbours i,j:  min(1, exp((beta_i - beta_j) (E_i - E_j)))  with E = -(b.s + sum J s s).
Derivation: the swap ratio is pi_i(x_j) pi_j(x_i) / (pi_i(x_i) pi_j(x_j)) with pi_t ~ exp(-beta_t E); the partition functions cancel.

Run from OUTSIDE master/:  cd /tmp && python -I <this file> [--rounds 300]   (about 2-4 min on CPU)
"""
import argparse
import itertools

import jax
import jax.numpy as jnp
import numpy as np
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

N_VIS, N_LAT = 8, 6
vis = [SpinNode() for _ in range(N_VIS)]
lat = [SpinNode() for _ in range(N_LAT)]
nodes = vis + lat
edges = [(v, h) for v in vis for h in lat]
free_blocks = [Block(vis), Block(lat)]


def make_model(seed, strength):
    rng = np.random.default_rng(seed)
    # prototype-like couplings: each latent unit locks onto a random +-1 visible pattern -> several deep basins
    J = strength * rng.choice([-1.0, 1.0], size=(N_VIS, N_LAT)) / np.sqrt(N_VIS)
    b = 0.05 * rng.standard_normal(N_VIS + N_LAT)
    return jnp.array(b, dtype=jnp.float32), jnp.array(J.reshape(-1), dtype=jnp.float32)


def exact_visible_law(b, J):
    S = np.array(list(itertools.product([-1, 1], repeat=N_VIS + N_LAT)), dtype=np.float32)
    ne = S @ np.array(b) + (S[:, :N_VIS].reshape(len(S), N_VIS, 1) * S[:, N_VIS:].reshape(len(S), 1, N_LAT)).reshape(len(S), -1) @ np.array(J)
    p = np.exp(ne - ne.max())
    p /= p.sum()
    vid = ((S[:, :N_VIS] > 0).astype(int) * (2 ** np.arange(N_VIS)[::-1])).sum(1)
    pv = np.bincount(vid, weights=p, minlength=2**N_VIS)
    mag = S[:, :N_VIS].mean(1)
    return pv, float((p * mag**2).sum())


def neg_energy_states(b, J, vis_s, lat_s):  # -E = b.s + sum J s s, batch over leading axis
    pair = jnp.einsum("...i,...j->...ij", vis_s, lat_s).reshape(*vis_s.shape[:-1], -1)
    return jnp.concatenate([vis_s, lat_s], -1) @ b + pair @ J


def spins(x):
    return 2.0 * x.astype(jnp.float32) - 1.0


def vis_id(vbool):
    return (np.asarray(vbool).astype(int) * (2 ** np.arange(N_VIS)[::-1])).sum(-1)


def split_rhat(x):  # x: (chains, draws)
    c, n = x.shape
    h = n // 2
    z = np.concatenate([x[:, :h], x[:, h : 2 * h]], 0)
    m, v = z.mean(1), z.var(1, ddof=1)
    W, B = v.mean(), h * m.var(ddof=1)
    return float(np.sqrt(((h - 1) / h * W + B / h) / W))


def ess_iat(x):
    """Multi-chain ESS (Gelman et al. BDA3 / Vehtari et al. 2021, without rank-normalisation).

    rho_t = 1 - (W - mean_m acov_m(t)) / var_plus, with W the within-chain variance and var_plus the pooled variance
    estimate that INCLUDES the between-chain variance; Geyer's initial positive pair sequence truncates the sum.
    A within-chain estimator (each chain centred on its own mean) removes the between-chain variance and reports an
    excellent ESS for chains trapped in different basins - the failure this script exists to expose."""
    m, n = x.shape
    xc = x - x.mean(1, keepdims=True)
    f = np.fft.rfft(xc, 2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n  # per-chain autocovariance (biased)
    acov_mean = acov.mean(0) * n / (n - 1)
    W = x.var(1, ddof=1).mean()
    var_plus = W * (n - 1) / n + (x.mean(1).var(ddof=1) if m > 1 else 0.0)
    rho = 1.0 - (W - acov_mean) / var_plus
    rho[0] = 1.0
    tau, k = -1.0, 0
    prev = np.inf
    while k + 1 < n:
        pair = rho[k] + rho[k + 1]
        if pair <= 0:
            break
        pair = min(pair, prev)  # monotone sequence
        prev = pair
        tau += 2 * pair
        k += 2
    tau = max(tau, 1.0 / np.log10(m * n))  # Vehtari et al. floor on tau
    return m * n / tau, tau


def tv(pv_exact, vids):
    emp = np.bincount(np.asarray(vids).ravel(), minlength=2**N_VIS) / np.asarray(vids).size
    return float(0.5 * np.abs(emp - pv_exact).sum())


def tv_floor(pv_exact, n, reps=50, seed=0):
    """TV of n INDEPENDENT exact draws to the exact law: the best any sampler can show at this sample size."""
    rng = np.random.default_rng(seed)
    pv64 = np.asarray(pv_exact, dtype=np.float64)
    pv64 = pv64 / pv64.sum()
    return float(np.mean([0.5 * np.abs(rng.multinomial(n, pv64) / n - pv_exact).sum() for _ in range(reps)]))


def report(name, mag, vids, pv_exact, m2_exact):
    r = split_rhat(mag)
    ess, tau = ess_iat(mag)
    print(f"{name:22s} R-hat {r:6.3f}  IAT {tau:7.1f}  ESS {ess:8.1f}/{mag.size}  TV(visible) {tv(pv_exact, vids):.3f} "
          f"(iid floor {tv_floor(pv_exact, mag.size):.3f})  E[m^2] {np.mean(mag**2):.3f} (exact {m2_exact:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--chains", type=int, default=32)
    ap.add_argument("--strength", type=float, default=7.0, help="coupling scale; larger = harder to mix")
    ap.add_argument("--seed", type=int, default=0, help="model seed")
    ap.add_argument("--key-seed", type=int, default=0, help="chain randomness (0 reproduces the first results)")
    ap.add_argument("--betas", default="1.0,0.55,0.3,0.15", help="PT ladder; aim for neighbour swap acceptance 0.2-0.8")
    ap.add_argument("--matched", action="store_true", help="also run plain Gibbs for len(betas)*rounds (compute-matched baseline)")
    a = ap.parse_args()
    b, J = make_model(a.seed, a.strength)
    pv_exact, m2_exact = exact_visible_law(b, J)
    C, R, k = a.chains, a.rounds, 5
    betas = [float(v) for v in a.betas.split(",")]

    def make_sweeper(beta):
        model = IsingEBM(nodes, edges, b, J, jnp.array(beta))
        prog = IsingSamplingProgram(model, free_blocks, [])
        sched = SamplingSchedule(n_warmup=k, n_samples=1, steps_per_sample=1)

        @jax.jit
        def sweep(state, key):
            keys = jax.random.split(key, C)
            out = jax.vmap(lambda kk, v, h: sample_states(kk, prog, sched, [v, h], [], free_blocks))(keys, state[0], state[1])
            return out[0][:, -1], out[1][:, -1]

        return model, sweep

    models, sweepers = zip(*[make_sweeper(bt) for bt in betas])
    key = jax.random.key(a.seed) if a.key_seed == 0 else jax.random.fold_in(jax.random.key(a.seed), a.key_seed)
    key, ki = jax.random.split(key)
    init = hinton_init(ki, models[0], free_blocks, (C,))

    # ---- plain block Gibbs at beta = 1, R*k sweeps per chain ------------------------------------
    def run_plain(rounds, key):
        st = init
        mags, vids = [], []
        for _ in range(rounds):
            key, kk = jax.random.split(key)
            st = sweepers[0](st, kk)
            mags.append(np.asarray(spins(st[0]).mean(-1)))
            vids.append(vis_id(st[0]))
        return np.stack(mags, 1), np.stack(vids, 1), key

    mg, vd, key = run_plain(R, key)
    report("plain Gibbs beta=1", mg, vd, pv_exact, m2_exact)
    if a.matched:
        mg, vd, key = run_plain(R * len(betas), key)
        report(f"plain Gibbs x{len(betas)} (matched)", mg, vd, pv_exact, m2_exact)

    # ---- parallel tempering: len(betas) replicas per system, swap neighbours every round ----------
    T = len(betas)
    states = [hinton_init(jax.random.fold_in(ki, t), models[0], free_blocks, (C,)) for t in range(T)]
    mags, vids, acc = [], [], np.zeros(T - 1)
    for r in range(R):
        for t in range(T):
            key, kk = jax.random.split(key)
            states[t] = sweepers[t](states[t], kk)
        for t in range(r % 2, T - 1, 2):  # alternate even / odd neighbour pairs
            ei = -neg_energy_states(b, J, spins(states[t][0]), spins(states[t][1]))
            ej = -neg_energy_states(b, J, spins(states[t + 1][0]), spins(states[t + 1][1]))
            logacc = (betas[t] - betas[t + 1]) * (ei - ej)
            key, ku = jax.random.split(key)
            do = jnp.log(jax.random.uniform(ku, (C,))) < logacc
            acc[t] += float(do.mean())
            new_t = [jnp.where(do[:, None], states[t + 1][i], states[t][i]) for i in range(2)]
            new_u = [jnp.where(do[:, None], states[t][i], states[t + 1][i]) for i in range(2)]
            states[t], states[t + 1] = new_t, new_u
        mags.append(np.asarray(spins(states[0][0]).mean(-1)))
        vids.append(vis_id(states[0][0]))
    report("parallel tempering", np.stack(mags, 1), np.stack(vids, 1), pv_exact, m2_exact)
    print(f"PT swap acceptance per neighbour pair (rounds counted when proposed): {np.round(acc / (R / 2), 2).tolist()}")
    print(f"cost: PT uses {T}x the sweeps of plain Gibbs for the same number of recorded draws (use --matched for the equal-compute baseline).")


if __name__ == "__main__":
    main()
