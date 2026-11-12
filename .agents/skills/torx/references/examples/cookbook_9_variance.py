"""Cookbook 9: variance of the three parameter-shift estimators, cost-normalised, plus REINFORCE bias.
Run from OUTSIDE master/.

Per-gate formulas (S = num_samples, p = sigmoid(theta_j), f = readout):
  param_shift_single (paper Eq. 40): sd = sqrt( (Var f at theta + Var f at theta^-) / S ),  sigma(theta^-) = p^2
  param_shift_inf   : g_j = p(1-p) (E1 - E0), E_b = mean of f with gate j forced to branch b
                      sd = p(1-p) sqrt( (Var_0 f + Var_1 f) / S )
  param_shift_filter: ratio estimator; single-gate closed form in part 1b/1c, delta-method bias/variance in cookbook_13 part 1e.
Circuit runs per gradient (sampled.py): inf (1 + G*K)*S, single (1 + G)*S, filter S  (G gates, K branches).
"""
import jax, jax.numpy as jnp, numpy as np
from torx import psc

sv = psc.StateVectorSimulator()
x_sv4, x_b2 = jnp.zeros(4).at[0].set(1.0), jnp.zeros(2, jnp.int32)
sig = jax.nn.sigmoid


def grads(circ, th, x_b, method, S, R, f_idx, seed=0):
    b = psc.BranchingSimulator(num_samples=S, diff_method=method)
    g = jax.jit(jax.grad(lambda z, k: b.expval_all(b.build_circuit(circ, z), x_b, k)[f_idx]))
    return np.array([np.concatenate([np.ravel(np.array(a)) for a in g(th, jax.random.key(seed + i))]) for i in range(R)])


def exact(circ, th, x_sv, f_idx):
    gr = jax.grad(lambda z: sv.expval_all(sv.build_circuit(circ, z), x_sv)[f_idx])(th)
    return np.concatenate([np.ravel(np.array(a)) for a in gr])


def prob_f(circ, th, x_sv, f_idx):
    return float(sv.expval_all(sv.build_circuit(circ, th), x_sv)[f_idx])


# ---- 1. single PNOT, theta=0 ------------------------------------------------------------------------
c1 = psc.DiscretePCircuit([psc.PNOT(0)])
th1, S, R = [jnp.array([0.0])], 2000, 300
p = 0.5
pred_single = np.sqrt((p * (1 - p) + p ** 2 * (1 - p ** 2)) / S)
print(f"1. single PNOT, theta=0, S={S}, R={R}: exact grad {exact(c1, th1, jnp.array([1.0, 0.0]), 0)[0]:.3f}")
for m in ("param_shift_inf", "param_shift_single", "param_shift_filter"):
    g = grads(c1, th1, jnp.zeros(1, jnp.int32), m, S, R, 0)
    extra = f"  (Eq. 40 predicts {pred_single:.4f}; _inf predicts 0)" if m != "param_shift_filter" else ""
    print(f"   {m:20s} mean {g.mean():.4f} sd {g.std():.4f}{extra}")

# ---- 1b. param_shift_filter, single gate: g = p (1 - p_hat)  ->  sd = p sqrt(p (1 - p) / S) -------------------------------
# (conditional means are exactly 1 and 0; the extra variance comes from the EMPIRICAL overall mean p_hat replacing p)
print("\n1b. single PNOT, param_shift_filter: predicted sd = p sqrt(p(1-p)/S) vs measured (S=2000, 400 repeats)")
for t in (-1.4, 0.0, 1.4):
    pp = float(sig(t))
    g = grads(c1, [jnp.array([t])], jnp.zeros(1, jnp.int32), "param_shift_filter", 2000, 400, 0, seed=700)
    print(f"   theta={t:+.1f} (p={pp:.2f}): predicted {pp * np.sqrt(pp * (1 - pp) / 2000):.5f}  measured {g.std():.5f}")

c_k3 = psc.DiscretePCircuit([psc.PditCycle(0, dims=3)])
x_k3_b = jnp.zeros(1, jnp.int32)
# ---- 1c. param_shift_filter, single K=3 gate: sd_j = p_j sqrt(Var f / S)  (K=2: Var f = p(1-p) -> the formula above) --------------------
print("\n1c. single PditCycle (K=3), readout = index, param_shift_filter, S=2000, 400 repeats: predicted sd_j = p_j sqrt(Var f / S) vs measured")
for thv in ([0.3, 0.1], [-1.0, 1.5]):
    th_ = [jnp.array(thv)]
    pj = np.array(jax.nn.softmax(jnp.array([0.0] + thv)))        # stay, forward, backward
    idx = np.array([0.0, 1.0, 2.0])
    Ef = float((pj * idx).sum()); Vf = float((pj * idx ** 2).sum() - Ef ** 2)
    g = grads(c_k3, th_, x_k3_b, "param_shift_filter", 2000, 400, 0, seed=800)
    pred = pj[1:] * np.sqrt(Vf / 2000)
    print(f"   theta={thv}: predicted [{', '.join('%.5f' % x for x in pred)}]  measured [{', '.join('%.5f' % x for x in g.std(0))}]")

# ---- 2. per-gate prediction on the 3-gate chain at theta=0 -----------------------------------------------
c3 = psc.DiscretePCircuit([psc.PNOT(0), psc.PCNOT([0, 1]), psc.PNOT(1)])
th0 = [jnp.array([0.0])] * 3
S, R = 1000, 300
meas = {m: grads(c3, th0, x_b2, m, S, R, 1, seed=100) for m in ("param_shift_inf", "param_shift_single")}
print(f"\n2. 3-gate chain, theta=0, S={S}, R={R}: per-gate sd, predicted vs measured")
for j in range(3):
    pj = 0.5
    # f variance with gate j forced to branch b: theta_j = +/-30 (deterministic)
    v = []
    for t in (-30.0, 30.0):
        th = [jnp.array([0.0])] * 3; th[j] = jnp.array([t]); q = prob_f(c3, th, x_sv4, 1); v.append(q * (1 - q))
    pred_inf = pj * (1 - pj) * np.sqrt((v[0] + v[1]) / S)
    thm = [jnp.array([0.0])] * 3
    q0 = prob_f(c3, thm, x_sv4, 1)
    # shifted theta^-: sigma(theta^-) = p^2  ->  theta^- = log(p^2/(1-p^2))
    thm[j] = jnp.array([float(jnp.log(pj ** 2 / (1 - pj ** 2)))])
    q1 = prob_f(c3, thm, x_sv4, 1)
    pred_single = np.sqrt((q0 * (1 - q0) + q1 * (1 - q1)) / S)
    print(f"   gate {j}: _inf  pred {pred_inf:.4f} meas {meas['param_shift_inf'][:, j].std():.4f}   "
          f"_single pred {pred_single:.4f} meas {meas['param_shift_single'][:, j].std():.4f}")

# ---- 3. equal total circuit runs T = 28,000 ---------------------------------------------------------------
G, K, T = 3, 2, 28000
ex = exact(c3, th0, x_sv4, 1)
print(f"\n3. equal budget T={T} circuit runs per gradient (3 gates, K=2), theta=0, mean sd over gates, R=100:")
plan = {"param_shift_inf": T // (1 + G * K), "param_shift_single": T // (1 + G), "param_shift_filter": T}
for m, S_m in plan.items():
    g = grads(c3, th0, x_b2, m, S_m, 100, 1, seed=500)
    print(f"   {m:20s} S={S_m:>6}  sd {g.std(0).mean():.4f}  max|bias| {np.abs(g.mean(0) - ex).max():.4f}")


# REINFORCE at the same budget (plain JAX surrogate from cookbook_6), baseline bias and the M/(M-1) correction
def run_chain(theta, key):
    k = jax.random.split(key, 3); p = sig(theta)
    b = jax.vmap(lambda kk, pp: jax.random.bernoulli(kk, pp))(k, p).astype(jnp.int32)
    z0 = b[0]; z1 = (z0 & b[1]) ^ b[2]
    return z1.astype(jnp.float32), jnp.sum(jnp.where(b == 1, jnp.log(p), jnp.log1p(-p)))

def make_surrogate(correct):
    def surrogate(theta, keys):
        f, logp = jax.vmap(lambda kk: run_chain(theta, kk))(keys)
        M = f.shape[0]
        base = jax.lax.stop_gradient(f.mean())
        scale = M / (M - 1) if correct else 1.0
        return scale * jnp.mean(jax.lax.stop_gradient(f - base) * logp)
    return surrogate

th = jnp.zeros(3)
gu = jax.jit(jax.grad(make_surrogate(False))); gc = jax.jit(jax.grad(make_surrogate(True)))
est = np.array([np.array(gu(th, jax.random.split(jax.random.key(7000 + i), T))) for i in range(100)])
print(f"   {'REINFORCE':20s} M={T:>6}  sd {est.std(0).mean():.4f}  max|bias| {np.abs(est.mean(0) - ex).max():.4f}")
M, R = 20, 4000
keys = [jax.random.split(jax.random.key(9000 + i), M) for i in range(R)]
bu = np.mean([np.array(gu(th, k)) for k in keys], 0) - ex
bc = np.mean([np.array(gc(th, k)) for k in keys], 0) - ex
print(f"   baseline bias at M={M} ({R} repeats): uncorrected {bu.round(4)}, with M/(M-1) {bc.round(4)}  "
      f"(gate 3: uncorrected/exact = {1 + bu[2] / ex[2]:.3f}, expected (M-1)/M = {(M - 1) / M:.3f})")

# ---- 4. param_shift_filter with an extreme p and tiny N ----------------------------------------------------
thx = [jnp.array([-4.0])] * 3
exx = exact(c3, thx, x_sv4, 1)
print(f"\n4. param_shift_filter at theta=-4 (p={float(sig(-4.0)):.3f}), exact grad {exx.round(4)}, R=1000:")
for N in (20, 200, 2000):
    g = grads(c3, thx, x_b2, "param_shift_filter", N, 1000, 1, seed=900)
    print(f"   N={N:>5}: mean {g.mean(0).round(4)}  bias {(g.mean(0) - exx).round(4)}  SE {(g.std(0, ddof=1) / np.sqrt(1000)).round(4)}")

# ---- 5. equal budget over a theta grid (RMSE per gate, mean over gates, 60 repeats) -------------------------
print(f"\n5. equal budget T={T}: RMSE (sqrt(bias^2 + sd^2), mean over the 3 gates) vs theta; lower is better")
print(f"   {'theta':>6} {'_inf':>9} {'_single':>9} {'_filter':>9}")
for t in (-4.0, -2.0, 0.0, 2.0):
    thg = [jnp.array([t])] * 3
    exg = exact(c3, thg, x_sv4, 1)
    row = []
    for m, S_m in plan.items():
        g = grads(c3, thg, x_b2, m, S_m, 60, 1, seed=1300)
        row.append(np.sqrt(np.mean((g - exg) ** 2)))
    print(f"   {t:>+6.0f} {row[0]:>9.5f} {row[1]:>9.5f} {row[2]:>9.5f}   best: {['_inf', '_single', '_filter'][int(np.argmin(row))]}")


# ---- 6. depth: _inf and _single formulas on longer chains (theta=0, S=1000, 200 repeats) ------------------
def predicted_sds(circ, th, x_sv, S, f_idx):
    out_inf, out_single = [], []
    for j in range(len(th)):
        pj = float(sig(th[j][0]))
        v = []
        for t in (-30.0, 30.0):
            z = list(th); z[j] = jnp.array([t]); q = prob_f(circ, z, x_sv, f_idx); v.append(q * (1 - q))
        out_inf.append(pj * (1 - pj) * np.sqrt((v[0] + v[1]) / S))
        z = list(th); z[j] = jnp.array([float(jnp.log(pj ** 2 / (1 - pj ** 2)))]); q1 = prob_f(circ, z, x_sv, f_idx)
        q0 = prob_f(circ, th, x_sv, f_idx)
        out_single.append(np.sqrt((q0 * (1 - q0) + q1 * (1 - q1)) / S))
    return np.array(out_inf), np.array(out_single)

print("\n6. depth: mean per-gate sd, predicted vs measured (theta=0, S=1000, 200 repeats)")
for depth in (6, 10):
    gates = [psc.PNOT(i % 2) if i % 3 != 1 else psc.PCNOT([0, 1]) for i in range(depth)]
    cd = psc.DiscretePCircuit(gates)
    thd = [jnp.array([0.0])] * depth
    pi, ps = predicted_sds(cd, thd, x_sv4, 1000, 1)
    mi = grads(cd, thd, x_b2, "param_shift_inf", 1000, 200, 1, seed=2000).std(0)
    ms = grads(cd, thd, x_b2, "param_shift_single", 1000, 200, 1, seed=2100).std(0)
    print(f"   depth {depth:>2}: _inf pred {pi.mean():.4f} meas {mi.mean():.4f}   _single pred {ps.mean():.4f} meas {ms.mean():.4f}")

# ---- 7. param_shift_filter with a K=3 gate and rare branches (PditCycle theta=[-4,-4]: forward and backward p=0.018 each) ----
c_k3 = psc.DiscretePCircuit([psc.PditCycle(0, dims=3)])
th_k3 = [jnp.array([-4.0, -4.0])]
x_k3_sv, x_k3_b = jnp.array([1.0, 0, 0]), jnp.zeros(1, jnp.int32)
ex_k3 = exact(c_k3, th_k3, x_k3_sv, 0)
print(f"\n7. param_shift_filter, K=3 gate, rare branches (p=0.018 each), exact grad of E[index] {ex_k3.round(4)}, R=1000:")
for N in (20, 200, 2000):
    g = grads(c_k3, th_k3, x_k3_b, "param_shift_filter", N, 1000, 0, seed=3000)
    print(f"   N={N:>5}: mean {g.mean(0).round(4)}  bias {(g.mean(0) - ex_k3).round(4)}  SE {(g.std(0, ddof=1) / np.sqrt(1000)).round(4)}")

print("\nsingle-gate _filter term p sqrt(p(1-p)/S) for the 3-gate chain at theta=0, S=1000: %.4f (compare the measured _filter sd in section 3)" % (0.5 * np.sqrt(0.25 / 1000)))

# ---- 8. is the single-gate _filter term a lower bound on longer chains?  (theta=0, S=1000, 200 repeats) ---------------------------
term = 0.5 * np.sqrt(0.25 / 1000)
print(f"\n8. param_shift_filter mean per-gate sd on chains vs the single-gate term {term:.4f} (S=1000, 200 repeats):")
for depth in (3, 6, 10):
    gates_ = [psc.PNOT(i % 2) if i % 3 != 1 else psc.PCNOT([0, 1]) for i in range(depth)]
    cd_ = psc.DiscretePCircuit(gates_)
    thd_ = [jnp.array([0.0])] * depth
    mf = grads(cd_, thd_, x_b2, "param_shift_filter", 1000, 200, 1, seed=4000).std(0)
    print(f"   depth {depth:>2}: measured mean sd {mf.mean():.4f} (min {mf.min():.4f}, max {mf.max():.4f})")
