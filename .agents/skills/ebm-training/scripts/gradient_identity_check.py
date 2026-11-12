"""Check the EBM maximum-likelihood gradient identity three ways (Song & Kingma 2021, Eq. 3-4).

  (1) autodiff of the exact log Z  ==  E_p[-grad E]                 (identity, Eq. 3-4)
  (2) autodiff of the exact KL(p_data || p_model(v)) of a latent-variable Boltzmann machine
      ==  the sampled contrast <.>_+ - <.>_-  returned by THRML estimate_kl_grad
  (3) one descent step along that sampled gradient lowers the exact KL (sign convention)
  (4) closed-form RBM free energy  F(v) = -b_v.v - sum_j log 2cosh(b_j + sum_i J_ij v_i)  ==  -log sum_h e^{-E(v,h)}

Run from OUTSIDE master/ (thrml/ shadows the package):  cd /tmp && python <this file>
Exit code 0 only if every check passes. Runtime about 30-60 s on CPU.
"""
import itertools
import sys

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from thrml import Block, SamplingSchedule, SpinNode
from thrml.models import IsingEBM, IsingTrainingSpec, estimate_kl_grad, hinton_init

N_VIS, N_LAT = 3, 2
BETA = 1.0
TOL_IDENTITY = 1e-5  # float32 autodiff vs expectation


def build():
    vis = [SpinNode() for _ in range(N_VIS)]
    lat = [SpinNode() for _ in range(N_LAT)]
    nodes = vis + lat
    edges = [(v, h) for v in vis for h in lat]  # complete bipartite RBM
    return vis, lat, nodes, edges


def enumerate_states(nodes, edges):
    """Exact-enumeration tables. Integer indices are fine here: this is test code, not the model."""
    idx = {n: i for i, n in enumerate(nodes)}
    S = np.array(list(itertools.product([-1, 1], repeat=len(nodes))), dtype=np.float32)
    ei = np.array([idx[a] for a, _ in edges])
    ej = np.array([idx[b] for _, b in edges])
    vis_id = ((S[:, :N_VIS] > 0).astype(int) * (2 ** np.arange(N_VIS)[::-1])).sum(1)
    return jnp.array(S), ei, ej, jnp.array(vis_id)


def neg_energy(S, ei, ej, b, J):
    return BETA * (S @ b + (S[:, ei] * S[:, ej]) @ J)


def exact_logZ(S, ei, ej, b, J):
    return jax.scipy.special.logsumexp(neg_energy(S, ei, ej, b, J))


def exact_kl(S, ei, ej, vis_id, p_data, b, J):
    ne = neg_energy(S, ei, ej, b, J)
    logZ = jax.scipy.special.logsumexp(ne)
    m = jnp.max(ne)
    pv = jax.ops.segment_sum(jnp.exp(ne - m), vis_id, num_segments=2**N_VIS)
    logpv = jnp.log(pv) + m - logZ
    safe = jnp.where(p_data > 0, p_data, 1.0)
    return jnp.sum(jnp.where(p_data > 0, p_data * (jnp.log(safe) - logpv), 0.0))


def main():
    vis, lat, nodes, edges = build()
    S, ei, ej, vis_id = enumerate_states(nodes, edges)
    key = jax.random.key(0)
    kb, kw, k1, k2 = jax.random.split(key, 4)
    b = 0.3 * jax.random.normal(kb, (len(nodes),))
    J = 0.5 * jax.random.normal(kw, (len(edges),))

    # ---- (1) grad log Z = E_p[-grad E] ------------------------------------------------------
    g_b, g_J = jax.grad(lambda b_, J_: exact_logZ(S, ei, ej, b_, J_), argnums=(0, 1))(b, J)
    p = jax.nn.softmax(neg_energy(S, ei, ej, b, J))
    m_b = BETA * (p[:, None] * S).sum(0)
    m_J = BETA * (p[:, None] * (S[:, ei] * S[:, ej])).sum(0)
    err1 = float(max(jnp.abs(g_b - m_b).max(), jnp.abs(g_J - m_J).max()))
    print(f"(1) |autodiff grad logZ - E_p[-grad E]|_max = {err1:.2e}   (tol {TOL_IDENTITY:.0e})")

    # ---- (2) sampled THRML gradient vs exact KL gradient --------------------------------------
    counts = np.array([4, 0, 3, 1, 0, 5, 2, 1])  # 16 data points; p_data = counts/16
    p_data = jnp.array(counts / counts.sum(), dtype=jnp.float32)
    patterns = np.array(list(itertools.product([0, 1], repeat=N_VIS)))
    data = jnp.array(np.repeat(patterns, counts, axis=0).astype(bool))  # (16, 3), True = +1

    gb_exact, gJ_exact = jax.grad(lambda b_, J_: exact_kl(S, ei, ej, vis_id, p_data, b_, J_), argnums=(0, 1))(b, J)

    sched_pos = SamplingSchedule(n_warmup=30, n_samples=400, steps_per_sample=2)
    sched_neg = SamplingSchedule(n_warmup=100, n_samples=200, steps_per_sample=3)
    free_blocks = [Block(vis), Block(lat)]
    pos_blocks = [Block(lat)]
    beta = jnp.array(BETA)

    def sampled_grad(b_, J_, key):
        model = IsingEBM(nodes, edges, b_, J_, beta)
        spec = IsingTrainingSpec(model, [Block(vis)], [], pos_blocks, free_blocks, sched_pos, sched_neg)
        ka, kb_, kc = jax.random.split(key, 3)
        init_pos = hinton_init(ka, model, pos_blocks, (1, data.shape[0]))
        init_neg = hinton_init(kb_, model, free_blocks, (3000,))
        gw, gb, _, _ = estimate_kl_grad(kc, spec, nodes, edges, [data], [], init_pos, init_neg)
        return gw, gb

    reps = jax.random.split(k1, 5)
    runs = [sampled_grad(b, J, kk) for kk in reps]
    gw_all = jnp.stack([r[0] for r in runs])
    gb_all = jnp.stack([r[1] for r in runs])
    gw_mc, gb_mc = gw_all.mean(0), gb_all.mean(0)
    sd_w = float((gw_all.std(0) / np.sqrt(len(runs))).max())  # standard error of the mean over 5 independent keys
    sd_b = float((gb_all.std(0) / np.sqrt(len(runs))).max())
    err_w = float(jnp.abs(gw_mc - gJ_exact).max())
    err_b = float(jnp.abs(gb_mc - gb_exact).max())
    scale = float(max(jnp.abs(gJ_exact).max(), jnp.abs(gb_exact).max()))
    print(f"(2) max|thrml - exact| weights {err_w:.4f} (4 s.e. = {4*sd_w:.4f})  biases {err_b:.4f} (4 s.e. = {4*sd_b:.4f})  grad scale {scale:.3f}")
    ok2 = err_w < max(4 * sd_w, 0.01) and err_b < max(4 * sd_b, 0.01)

    # ---- (3) sign convention: descend along the sampled gradient lowers the exact KL ----------
    kl0 = float(exact_kl(S, ei, ej, vis_id, p_data, b, J))
    lr = 0.5
    kl1 = float(exact_kl(S, ei, ej, vis_id, p_data, b - lr * gb_mc, J - lr * gw_mc))
    kl_wrong = float(exact_kl(S, ei, ej, vis_id, p_data, b + lr * gb_mc, J + lr * gw_mc))
    print(f"(3) exact KL: before {kl0:.4f} | descend {kl1:.4f} | ascend {kl_wrong:.4f}")

    # ---- (4) closed-form free energy of the spin RBM (hiddens summed out analytically) --------
    ne = np.array(neg_energy(S, ei, ej, b, J))
    Sn, vid = np.array(S), np.array(vis_id)
    Jm = np.array(J).reshape(N_VIS, N_LAT)  # edges were built v-major
    err4 = 0.0
    for k in range(2**N_VIS):
        v = Sn[vid == k][0, :N_VIS]
        f_enum = -float(jax.scipy.special.logsumexp(ne[vid == k]))
        f_closed = -float(np.array(b)[:N_VIS] @ v) - float(np.log(2 * np.cosh(np.array(b)[N_VIS:] + v @ Jm)).sum())
        err4 = max(err4, abs(f_enum - f_closed))
    print(f"(4) |F_enumerated - F_closed_form|_max = {err4:.2e}   (tol 1e-4)")

    ok = err1 < TOL_IDENTITY and ok2 and kl1 < kl0 < kl_wrong and err4 < 1e-4
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
