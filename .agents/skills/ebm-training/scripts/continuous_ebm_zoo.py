"""Train the same small continuous EBM five ways and score it against the exact density.

Methods (Song & Kingma 2021):  cd_k (Sec. 3, from data) | mle_cd (persistent + replay buffer)  |  dsm (Sec. 4.1, Eq. 10)
                               ssm (Sec. 4.2, Eq. 17)                    |  nce (Sec. 5, Eq. 21, learnable log Z)

Target: two well-separated Gaussians with unequal weights (0.8 / 0.2). Score matching is blind to the
mixture weight when the modes are nearly disjoint (Sec. 4.4), so the metric that matters is the
recovered weight of the small mode, computed from the exact normalised model on a grid.

Run from OUTSIDE master/:  cd /tmp && python <this file> [--steps 3000] [--quick]
Prints a table; exit code 0. This is a diagnostic harness, not a pass/fail test, because the whole
point is to see which methods fail.
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

MU = jnp.array([[-2.0, 0.0], [2.0, 0.0]])
W = jnp.array([0.8, 0.2])
SIG = 0.45
LO, HI, NG = -5.0, 5.0, 220
LANG_STEPS, LANG_EPS, LANG_NOISE = 30, 0.3, 1.0
SSM_SIGN = -1.0  # sign of the Hessian term for an ENERGY; +1.0 reproduces the printed (wrong) Eq. 17
NCE_SIGMA = 2.5  # std of the isotropic Gaussian noise distribution
RESTART = 0.0  # fraction of replay-buffer chains restarted from uniform noise each step (0.05 collapsed in our runs)


def sample_data(key, n):
    k1, k2 = jax.random.split(key)
    comp = jax.random.bernoulli(k1, W[1], (n,)).astype(int)
    return MU[comp] + SIG * jax.random.normal(k2, (n, 2))


def true_logp(x):
    d = x[:, None, :] - MU[None]
    lg = -0.5 * (d**2).sum(-1) / SIG**2 - jnp.log(2 * jnp.pi * SIG**2)
    return jax.scipy.special.logsumexp(lg + jnp.log(W), axis=1)


def init_mlp(key, sizes=(2, 128, 128, 1)):
    ps = []
    for k, (i, o) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        ps.append((jax.random.normal(k, (i, o)) * jnp.sqrt(1.0 / i), jnp.zeros(o)))
    return ps


def energy(ps, x):
    h = x
    for W_, b_ in ps[:-1]:
        h = jax.nn.softplus(h @ W_ + b_)  # smooth: second derivatives exist for SM / SSM
    W_, b_ = ps[-1]
    return (h @ W_ + b_)[..., 0]


def score(ps, x):  # grad_x log p = -grad_x E (Eq. 5)
    return -jax.vmap(jax.grad(lambda xi: energy(ps, xi)))(x)


# ------------------------------- losses -------------------------------------------------------
def loss_dsm(ps, x, key, sigma=0.3):
    z = jax.random.normal(key, x.shape)
    xt = x + sigma * z
    target = -z / sigma  # grad log q(xt|x)
    return 0.5 * jnp.mean(jnp.sum((score(ps, xt) - target) ** 2, -1)) * sigma**2  # sigma^2 weighting


def loss_ssm(ps, x, key):
    v = jax.random.normal(key, x.shape)

    def one(xi, vi):
        g = lambda y: jax.grad(lambda yy: energy(ps, yy))(y)
        gv, hv = jax.jvp(g, (xi,), (vi,))  # H v
        return 0.5 * jnp.sum(g(xi) ** 2) + SSM_SIGN * (vi @ hv)  # Eq. 17; SSM_SIGN=-1 is correct for an energy

    return jnp.mean(jax.vmap(one)(x, v))


def loss_nce(ps_logz, x, key, nu=1.0):
    ps, logz = ps_logz
    n = x.shape[0]
    xn = NCE_SIGMA * jax.random.normal(key, (n, 2))  # broad Gaussian noise, known density
    logpn = lambda y: -0.5 * (y**2).sum(-1) / NCE_SIGMA**2 - jnp.log(2 * jnp.pi * NCE_SIGMA**2)
    logpm = lambda y: -energy(ps, y) - logz
    lo = lambda y: logpm(y) - logpn(y) + jnp.log(nu)  # log nu p_theta / p_n  (Eq. 18-19)
    return jnp.mean(jax.nn.softplus(-lo(x))) + nu * jnp.mean(jax.nn.softplus(lo(xn)))


def langevin(ps, x, key, steps, eps, noise_scale=1.0):
    # Eq. 6: x <- x - (eps^2/2) grad E + eps z.  noise_scale < 1 is the 'decoupled' trick (biased, mode-seeking).
    def body(c, k):
        g = jax.vmap(jax.grad(lambda xi: energy(ps, xi)))(c)
        c = c - 0.5 * eps**2 * g + noise_scale * eps * jax.random.normal(k, c.shape)
        return jnp.clip(c, -6, 6), None

    out, _ = jax.lax.scan(body, x, jax.random.split(key, steps))
    return out


CD_REG = 0.0


def loss_cd(ps, x, xneg, reg=None):
    reg = CD_REG if reg is None else reg
    ep, en = energy(ps, x), energy(ps, xneg)
    return jnp.mean(ep) - jnp.mean(en) + reg * jnp.mean(ep**2 + en**2)  # energy L2 reg, Du & Mordatch 2019


# ------------------------------- training -----------------------------------------------------
def train(method, steps, seed, bs=256):
    key = jax.random.key(seed)
    key, k0 = jax.random.split(key)
    ps = init_mlp(k0)
    state = (ps, jnp.array(0.0)) if method == "nce" else ps
    opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(optax.cosine_decay_schedule(1e-3, steps)))
    ost = opt.init(state)
    key, k_buf = jax.random.split(key)
    buf = jax.random.uniform(k_buf, (4096, 2), minval=-4, maxval=4)

    @jax.jit
    def step(state, ost, key, buf):
        kd, kl, kb, kr, kn = jax.random.split(key, 5)
        x = sample_data(kd, bs)
        if method == "dsm":
            l, g = jax.value_and_grad(loss_dsm)(state, x, kl)
        elif method == "ssm":
            l, g = jax.value_and_grad(loss_ssm)(state, x, kl)
        elif method == "nce":
            l, g = jax.value_and_grad(loss_nce)(state, x, kl)
        elif method == "cd_k":  # Hinton CD: chains start AT the data, k short Langevin steps, no persistence
            xneg = jax.lax.stop_gradient(langevin(state, x, kl, steps=LANG_STEPS, eps=LANG_EPS, noise_scale=LANG_NOISE))
            l, g = jax.value_and_grad(loss_cd)(state, x, xneg)
        else:  # mle_cd = persistent chains + replay buffer; 95% of chains continue, 5% restart from noise
            idx = jax.random.randint(kb, (bs,), 0, buf.shape[0])
            start = buf[idx]
            fresh = jax.random.uniform(kr, (bs, 2), minval=-4, maxval=4)
            start = jnp.where(jax.random.bernoulli(kn, RESTART, (bs, 1)), fresh, start)
            xneg = langevin(state, start, kl, steps=LANG_STEPS, eps=LANG_EPS, noise_scale=LANG_NOISE)
            xneg = jax.lax.stop_gradient(xneg)
            l, g = jax.value_and_grad(loss_cd)(state, x, xneg)
            buf = buf.at[idx].set(xneg)
        upd, ost = opt.update(g, ost, state)
        return optax.apply_updates(state, upd), ost, l, buf

    for i in range(steps):
        key, k = jax.random.split(key)
        state, ost, l, buf = step(state, ost, k, buf)
    return state, float(l)


# ------------------------------- exact evaluation --------------------------------------------
def evaluate(state, method):
    ps = state[0] if method == "nce" else state
    g = jnp.linspace(LO, HI, NG)
    X = jnp.stack(jnp.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    dA = (g[1] - g[0]) ** 2
    logu = -energy(ps, X)  # log unnormalised
    logZ = jax.scipy.special.logsumexp(logu) + jnp.log(dA)
    logp = logu - logZ
    pt = jnp.exp(true_logp(X))
    kl = float(jnp.sum(pt * (true_logp(X) - logp)) * dA)  # KL(true || model) on the grid
    pm = jnp.exp(logp)
    w_small = float(jnp.sum(jnp.where(X[:, 0] > 0, pm, 0.0)) * dA)  # mass in the right-hand mode
    out = dict(kl=kl, w_small=w_small)
    if method == "nce":
        out["logZ_learned"] = float(state[1])
        out["logZ_exact"] = float(logZ)
    return out


def main():
    global CD_REG, RESTART, SSM_SIGN, NCE_SIGMA, LANG_STEPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=1, help="run seeds 0..n-1 and report mean +- sd")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--methods", default="cd_k,mle_cd,dsm,ssm,nce")
    ap.add_argument("--reg", type=float, default=CD_REG, help="energy L2 weight for cd_k / mle_cd")
    ap.add_argument("--restart", type=float, default=RESTART, help="replay-buffer restart fraction (mle_cd)")
    ap.add_argument("--ssm-sign", type=float, default=SSM_SIGN, help="-1 correct for an energy, +1 = printed Eq. 17")
    ap.add_argument("--nce-sigma", type=float, default=NCE_SIGMA)
    ap.add_argument("--lang-steps", type=int, default=LANG_STEPS, help="Langevin steps per update for cd_k / mle_cd")
    a = ap.parse_args()
    CD_REG, RESTART, SSM_SIGN, NCE_SIGMA, LANG_STEPS = a.reg, a.restart, a.ssm_sign, a.nce_sigma, a.lang_steps
    steps = 600 if a.quick else a.steps
    print(f"target weights {np.array(W)}  sigma {SIG}  steps {steps}  seeds {a.seeds}  reg {CD_REG} restart {RESTART} "
          f"ssm_sign {SSM_SIGN} nce_sigma {NCE_SIGMA} lang_steps {LANG_STEPS}")
    print(f"{'method':8s} {'KL(true||model) mean+-sd':>26s} {'mass right mode mean+-sd':>26s}  note")
    for m in a.methods.split(","):
        kls, ws, notes = [], [], ""
        for seed in range(a.seeds):
            st, l = train(m, steps, seed)
            r = evaluate(st, m)
            kls.append(r["kl"])
            ws.append(r["w_small"])
            if m == "nce":
                notes = f"logZ learned {r['logZ_learned']:.2f} vs exact {r['logZ_exact']:.2f} (last seed)"
        kls, ws = np.array(kls), np.array(ws)
        print(f"{m:8s} {kls.mean():12.4f} +- {kls.std():8.4f} {ws.mean():14.3f} +- {ws.std():7.3f}  "
              f"per-seed mass {np.round(ws, 3).tolist()} {notes}", flush=True)
    print("true mass right mode = 0.200")


if __name__ == "__main__":
    main()
