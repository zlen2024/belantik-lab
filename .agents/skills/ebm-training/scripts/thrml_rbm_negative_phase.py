"""Negative-phase study for a latent Boltzmann machine (RBM) trained with THRML, scored against the EXACT KL.

Paper link: Song & Kingma 2021 Sec. 3 - maximum likelihood needs samples from the model; Contrastive Divergence
(CD, chains started at data, few steps) and Persistent CD (PCD, chains carried across updates) are cheaper but
their gradient is biased. THRML packages the positive/negative phases in estimate_kl_grad, so here only the
negative-phase INITIALISATION and LENGTH change.

  cd1 / cd10  : negative chains start at (v=data, h~p(h|data)), k Gibbs sweeps  [NB: free blocks are sampled in list
                order, so starting with `vis` first would overwrite the data; see sample_hidden_given_visible]
  pcd         : 1 sweep per update from persistent chains (state carried over)
  long        : fresh chains, 100 sweeps (close to the exact gradient; cost reference)

Outputs: exact KL(p_data || p_model(v)) over training, then the gradient-bias table at one fixed parameter
point (relative L2 error of the mean estimated gradient vs the exact gradient, next to its Monte Carlo floor).

Run from OUTSIDE master/:   cd /tmp && python -I <this file> [--steps 600] [--seeds 3] [--quick]
Needs: jax, optax, thrml. Runtime about 5-8 min on CPU for 3 seeds.
"""
import argparse
import itertools
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingTrainingSpec, estimate_kl_grad, hinton_init

N_VIS, N_LAT = 8, 6
BETA = jnp.array(1.0)
BATCH, N_CHAINS = 128, 128

vis = [SpinNode() for _ in range(N_VIS)]
lat = [SpinNode() for _ in range(N_LAT)]
nodes = vis + lat
edges = [(v, h) for v in vis for h in lat]
free_blocks = [Block(vis), Block(lat)]
pos_blocks = [Block(lat)]


# ---------------- exact tables (test code: integer indices are fine here) ---------------------------------
S_ALL = np.array(list(itertools.product([-1, 1], repeat=len(nodes))), dtype=np.float32)
IDX = {n: i for i, n in enumerate(nodes)}
EI = np.array([IDX[a] for a, _ in edges])
EJ = np.array([IDX[b] for _, b in edges])
VIS_ID = jnp.array(((S_ALL[:, :N_VIS] > 0).astype(int) * (2 ** np.arange(N_VIS)[::-1])).sum(1))
S_J = jnp.array(S_ALL)
PAIR = jnp.array(S_ALL[:, EI] * S_ALL[:, EJ])


def make_data_distribution(seed=0):
    """Four prototype patterns, each bit flipped with prob 0.08. Returns exact p_data over the 2^8 visible states."""
    rng = np.random.default_rng(seed)
    protos = rng.integers(0, 2, size=(4, N_VIS))
    pats = np.array(list(itertools.product([0, 1], repeat=N_VIS)))
    flip = 0.08
    p = np.zeros(len(pats))
    for pr in protos:
        d = (pats != pr).sum(1)
        p += 0.25 * flip**d * (1 - flip) ** (N_VIS - d)
    return pats, jnp.array(p / p.sum(), dtype=jnp.float32)


PATS, P_DATA = make_data_distribution()


def neg_energy(b, J):
    return S_J @ b + PAIR @ J


def exact_kl(b, J):
    ne = neg_energy(b, J)
    logZ = jax.scipy.special.logsumexp(ne)
    m = ne.max()
    pv = jax.ops.segment_sum(jnp.exp(ne - m), VIS_ID, num_segments=2**N_VIS)
    logpv = jnp.log(pv) + m - logZ
    return jnp.sum(P_DATA * (jnp.log(P_DATA) - logpv))


exact_grad = jax.jit(jax.grad(exact_kl, argnums=(0, 1)))
exact_kl_j = jax.jit(exact_kl)


def sample_data(key, n):
    idx = jax.random.choice(key, 2**N_VIS, (n,), p=P_DATA)
    return jnp.array(PATS)[idx].astype(bool)


# ---------------- training machinery ------------------------------------------------------------------------
SCHED_POS = SamplingSchedule(n_warmup=20, n_samples=1, steps_per_sample=1)


def sample_hidden_given_visible(b, J, data, key):
    """h ~ p(h | v) for the spin RBM: P(h_j=+1) = sigmoid(2 beta (b_j + sum_i J_ij v_i)). Edges are v-major."""
    v = 2.0 * data.astype(jnp.float32) - 1.0
    logits = 2.0 * BETA * (b[N_VIS:] + v @ J.reshape(N_VIS, N_LAT))
    return jax.random.bernoulli(key, jax.nn.sigmoid(logits))[...]


def make_grad_fn(mode):
    k = {"cd1": 1, "cd10": 10, "pcd": 1, "long": 100}[mode]
    sched_neg = SamplingSchedule(n_warmup=k, n_samples=1, steps_per_sample=1)

    def grad_fn(b, J, data, pers, key):
        model = IsingEBM(nodes, edges, b, J, BETA)
        spec = IsingTrainingSpec(model, [Block(vis)], [], pos_blocks, free_blocks, SCHED_POS, sched_neg)
        k_ip, k_in, k_g = jax.random.split(key, 3)
        init_pos = hinton_init(k_ip, model, pos_blocks, (1, data.shape[0]))
        if mode in ("cd1", "cd10"):
            # THRML samples free blocks in LIST order ([vis, lat]): the first half-sweep resamples vis from lat0 and
            # would overwrite `data`. So the chain must start at (data, h0 ~ p(h | data)); then sweep 1 gives
            # v1 ~ p(v|h0), h1 ~ p(h|v1), the textbook CD-1 pair. (Initialising lat from biases = a short run from noise.)
            lat0 = sample_hidden_given_visible(b, J, data, k_in)
            init_neg = [data, lat0]
        elif mode == "pcd":
            init_neg = pers
        else:
            init_neg = hinton_init(k_in, model, free_blocks, (N_CHAINS,))
        gw, gb, _, (mb_neg, _) = estimate_kl_grad(k_g, spec, nodes, edges, [data], [], init_pos, init_neg)
        new_pers = pers
        if mode == "pcd":  # reproduce the per-chain keys estimate_kl_grad used, so we get the SAME chain's end state
            _, key_neg = jax.random.split(k_g, 2)
            keys_neg = jax.random.split(key_neg, init_neg[0].shape[0])
            ends = jax.vmap(lambda kk, ii: sample_states(kk, spec.program_negative, sched_neg, ii, [], free_blocks))(
                keys_neg, init_neg
            )
            new_pers = [e[:, -1] for e in ends]
            check = jnp.abs(mb_neg[:, :N_VIS] - (2.0 * new_pers[0] - 1.0)).max()  # must be 0
            return gw, gb, new_pers, check
        return gw, gb, new_pers, jnp.array(0.0)

    return jax.jit(grad_fn)


def train(mode, steps, seed, lr=0.02):
    key = jax.random.key(seed)
    key, kb, kw, kp = jax.random.split(key, 4)
    b = 0.01 * jax.random.normal(kb, (len(nodes),))
    J = 0.05 * jax.random.normal(kw, (len(edges),))
    opt = optax.adam(lr)
    ost = opt.init((b, J))
    grad_fn = make_grad_fn(mode)
    pers = hinton_init(kp, IsingEBM(nodes, edges, b, J, BETA), free_blocks, (N_CHAINS,))
    curve, worst_check = [], 0.0
    for t in range(steps + 1):
        if t % max(1, steps // 6) == 0:
            curve.append((t, float(exact_kl_j(b, J))))
        key, kd, kg = jax.random.split(key, 3)
        data = sample_data(kd, BATCH)
        gw, gb, pers, chk = grad_fn(b, J, data, pers, kg)
        worst_check = max(worst_check, float(chk))
        upd, ost = opt.update((gb, gw), ost, (b, J))
        b, J = optax.apply_updates((b, J), upd)
    return (b, J), curve, worst_check


def gradient_bias(bJ, repeats, seed=123, n_chains=2048):
    """Mean over `repeats` independent gradient estimates vs the exact gradient at one parameter point.
    All estimators use the same number of negative chains (n_chains = data batch = 2048) so their floors are comparable."""
    global N_CHAINS
    N_CHAINS = n_chains
    b, J = bJ
    gb_ex, gw_ex = exact_grad(b, J)
    g_ex = jnp.concatenate([gb_ex, gw_ex])
    rows = []
    for mode in ["cd1", "cd10", "pcd", "long"]:
        fn = make_grad_fn(mode)
        key = jax.random.key(seed)
        pers = hinton_init(jax.random.key(7), IsingEBM(nodes, edges, b, J, BETA), free_blocks, (N_CHAINS,))
        if mode == "pcd":  # burn the persistent chains in at this fixed theta so they are near-stationary
            sched = SamplingSchedule(n_warmup=300, n_samples=1, steps_per_sample=1)
            prog = IsingTrainingSpec(IsingEBM(nodes, edges, b, J, BETA), [Block(vis)], [], pos_blocks, free_blocks, sched, sched)
            keys = jax.random.split(jax.random.key(9), N_CHAINS)
            ends = jax.vmap(lambda kk, ii: sample_states(kk, prog.program_negative, sched, ii, [], free_blocks))(keys, pers)
            pers = [e[:, -1] for e in ends]
        gs = []
        for r in range(repeats):
            key, kd, kg = jax.random.split(key, 3)
            data = sample_data(kd, 2048)
            gw, gb, pers, _ = fn(b, J, data, pers, kg)
            gs.append(jnp.concatenate([gb, gw]))
        gs = jnp.stack(gs)
        mean = gs.mean(0)
        rel_bias = float(jnp.linalg.norm(mean - g_ex) / jnp.linalg.norm(g_ex))
        floor = float(jnp.linalg.norm(gs.std(0) / np.sqrt(repeats)) / jnp.linalg.norm(g_ex))
        cos = float(mean @ g_ex / (jnp.linalg.norm(mean) * jnp.linalg.norm(g_ex)))
        rows.append((mode, rel_bias, floor, cos))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=24)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-train", action="store_true", help="only the gradient-bias table")
    a = ap.parse_args()
    steps, reps, n_seeds = (150, 8, 1) if a.quick else (a.steps, a.repeats, a.seeds)
    h = float(-(P_DATA * jnp.log(P_DATA)).sum())
    print(f"RBM {N_VIS} visible + {N_LAT} latent, exact enumeration over {2**len(nodes)} states; H(p_data) = {h:.3f} nats")
    print(f"exact KL(p_data||p_model) every {max(1, steps // 6)} steps, {steps} Adam steps, batch {BATCH}, {n_seeds} seeds")
    modes = ["cd1", "cd10", "pcd", "long"]
    finals = {m: [] for m in modes}
    for seed in range(0 if a.skip_train else n_seeds):
        for mode in modes:
            t0 = time.time()
            bJ, curve, chk = train(mode, steps, seed)
            finals[mode].append(curve[-1][1])
            cs = "  ".join(f"{t}:{kl:.3f}" for t, kl in curve)
            extra = f"  key-reproduction check |dmoment|={chk:.1e}" if mode == "pcd" else ""
            print(f"seed {seed} {mode:5s} KL {cs}   [{time.time()-t0:.0f}s]{extra}")
    print("final exact KL, mean +- sd (ddof=0) over seeds:")
    for m in ([] if a.skip_train else modes):
        v = np.array(finals[m])
        print(f"  {m:5s} {v.mean():.3f} +- {v.std():.3f}   (per seed {np.round(v, 3).tolist()})")

    bias_pt, _, _ = train("long", 40, 0)  # mid-training point: the exact gradient is large, so ratios are meaningful
    kl_pt = float(exact_kl_j(*bias_pt))
    print(f"Gradient bias at a mid-training point (40 'long' steps, exact KL {kl_pt:.3f}); exact gradient = autodiff of the enumerated KL.")
    print("excess = sqrt(max(bias^2 - floor^2, 0)): bias beyond Monte Carlo noise. The floor assumes independent repeats, so it is")
    print("valid for cd1/cd10/long (fresh chains each repeat) and optimistic for pcd (successive estimates share a chain).")
    print(f"{'estimator':9s} {'rel.bias':>9s} {'MC floor':>9s} {'excess':>8s} {'cos':>7s}")
    for mode, rb, fl, cs in gradient_bias(bias_pt, reps):
        print(f"{mode:9s} {rb:9.3f} {fl:9.3f} {np.sqrt(max(rb**2 - fl**2, 0.0)):8.3f} {cs:7.3f}")


if __name__ == "__main__":
    main()
