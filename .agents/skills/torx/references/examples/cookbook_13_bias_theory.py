"""Cookbook 13: theory of the nonlinear-loss bias, fixed points, and the energy-based gradient (paper Eq. 44).
Run from OUTSIDE master/.

1. bias of param_shift_single for a loss L(E_hat), per gate j (second order in the sample fluctuation):
       bias_j = [ L''(E) + 1/2 L'''(E) (E - E'_j) ] Var(f) / S
   E'_j = expectation of the readout with gate j moved to theta_j^- (sigma(theta^-) = sigma(theta)^2), the shifted circuit of the estimator.
   Quadratic loss: L''' = 0 -> 2 Var(f)/S on every gate. Checked with independent keys, R = 20,000.
2. Stationary points on a single PNOT with plain SGD, loss (E_hat - t)^2:  _inf: t   _single: t - 1/S   _filter: t/(1 - 1/S)
   and on the 3-gate chain (deterministic prediction from the exact gradient + the law, against the sampled Adam run).
3. Paper Eq. 44 (energy-based kernel gradient): a toy gate, then a gate with a downstream gate, with a baseline.
"""
import jax, jax.numpy as jnp, numpy as np
import optax
from torx import psc

sv = psc.StateVectorSimulator()
x_sv, x_b = jnp.zeros(4).at[0].set(1.0), jnp.zeros(2, jnp.int32)
fmt = lambda v: " ".join("%.4f" % x for x in np.asarray(v, dtype=float))

# ---- 1. per-gate law ---------------------------------------------------------------------------------------------------------
circuits = {
    "4-gate": (psc.DiscretePCircuit([psc.PNOT(0), psc.PCNOT([0, 1]), psc.PNOT(0), psc.PNOT(1)]), jnp.array([0.3, -0.4, 0.5, 0.8])),
    "6-gate mixed": (psc.DiscretePCircuit([psc.PNOT(0), psc.PReset(1), psc.PCNOT([0, 1]), psc.PSWAP([0, 1]), psc.PNOT(1), psc.POR([0, 1])]),
                     jnp.array([0.4, -0.3, 0.7, 0.5, -0.2, 0.9])),
}
READ = {"bit 1": np.array([0.0, 1.0]), "sum of bits": np.array([1.0, 1.0]), "bit0 - bit1": np.array([1.0, -1.0])}
LOSS = {"quad": lambda t: (lambda e: (e - t) ** 2), "cubic": lambda t: (lambda e: e ** 3), "exp": lambda t: (lambda e: jnp.exp(-3.0 * e))}


def n_of(circ):
    return len(circ.gates)


def E_and_var(circ, th, w):
    n = n_of(circ)
    p = np.array(sv.density(sv.build_circuit(circ, [th[i:i + 1] for i in range(n)]), x_sv))
    f = np.array([0.0, w[1], w[0], w[0] + w[1]])               # states 00, 01, 10, 11 (site 0 = MSB)
    E = float((p * f).sum()); return E, float((p * f ** 2).sum() - E ** 2)


def shifted_E(circ, th, w, j):
    p = float(jax.nn.sigmoid(th[j]))
    th2 = th.at[j].set(np.log(p ** 2 / (1 - p ** 2)))
    return E_and_var(circ, th2, w)[0]


def terms(circ, th, w, L):
    E, V = E_and_var(circ, th, w)
    L2 = float(jax.grad(jax.grad(L))(E)); L3 = float(jax.grad(jax.grad(jax.grad(L)))(E))
    g = np.array([E - shifted_E(circ, th, w, j) for j in range(n_of(circ))])      # dE/dtheta_j = E - E'_j
    return E, V, L2, L3, g


def predicted(circ, th, w, L, S, method="param_shift_single"):
    E, V, L2, L3, g = terms(circ, th, w, L)
    cov = L2 if method == "param_shift_single" else 0.0     # _inf has no covariance term, only the Jensen term
    return (cov + 0.5 * L3 * g) * V / S


def measure(circ, th, w, L, S, R=20000, seed=0, method="param_shift_single"):
    n = n_of(circ)
    b = psc.BranchingSimulator(num_samples=S, diff_method=method)
    wj = jnp.asarray(w)
    Ehat = lambda th_, k: jnp.dot(wj, b.expval_all(b.build_circuit(circ, [th_[i:i + 1] for i in range(n)]), x_b, k))
    g = jax.jit(jax.vmap(jax.grad(lambda th_, k: L(Ehat(th_, k))), in_axes=(None, 0)))
    est = np.concatenate([np.array(g(th, jax.random.split(jax.random.key(seed + c), R // 4))) for c in range(4)])
    Ex = lambda th_: jnp.dot(wj, sv.expval_all(sv.build_circuit(circ, [th_[i:i + 1] for i in range(n)]), x_sv))
    true = np.array(jax.grad(lambda th_: L(Ex(th_)))(th))
    return est.mean(0) - true, est.std(0) / np.sqrt(len(est))


print("1. bias of param_shift_single per gate: predicted [L'' + L'''(E - E'_j)/2] Var/S vs measured (20,000 repeats, independent keys)")
cases = [("4-gate", "bit 1", "quad", 0.8, (40, 160)), ("6-gate mixed", "bit 1", "quad", 0.8, (40,)),
         ("6-gate mixed", "sum of bits", "quad", 1.0, (40, 160)), ("6-gate mixed", "bit0 - bit1", "quad", 0.2, (40, 160)),
         ("6-gate mixed", "bit 1", "cubic", 0.0, (40, 160)), ("6-gate mixed", "bit 1", "exp", 0.0, (40,))]
small_cases = [("6-gate mixed", "bit 1", "cubic", 0.0, (20,)), ("6-gate mixed", "bit 1", "exp", 0.0, (20,))]
seed = 100
for cname, rname, lname, t, Ss in cases:
    circ, th = circuits[cname]; w = READ[rname]; L = LOSS[lname](t)
    E, V = E_and_var(circ, th, w)
    for S in Ss:
        seed += 10
        pred = predicted(circ, th, w, L, S)
        bias, se = measure(circ, th, w, L, S, seed=seed)
        _E, _V, L2, L3, gg = terms(circ, th, w, L)
        print(f"   {cname:12s} {rname:12s} {lname:5s} S={S:>3} E={E:.3f} Var={V:.3f}  predicted [{fmt(pred)}]  measured [{fmt(bias)}]  SE up to {se.max():.4f}  "
              f"terms L2={L2:.3f} L3={L3:.3f} dE/dtheta_j=[{fmt(gg)}]")

print("\n1b. small S (second-order expansion loses accuracy): predicted vs measured per gate, 20,000 repeats")
for cname, rname, lname, t, Ss in small_cases:
    circ, th = circuits[cname]; w = READ[rname]; L = LOSS[lname](t)
    for S in Ss:
        seed += 10
        pred = predicted(circ, th, w, L, S); bias, se = measure(circ, th, w, L, S, seed=seed)
        print(f"   {cname:12s} {rname:12s} {lname:5s} S={S:>3}  predicted [{fmt(pred)}]  measured [{fmt(bias)}]  SE up to {se.max():.4f}  ratio measured/predicted {fmt(bias / pred)}")

print("\n1c. the same loss with param_shift_inf (bias = [L3(E)/2] dE/dtheta_j Var/S, no covariance term) and param_shift_filter (measured only), 6-gate mixed circuit, bit 1")
circ, th = circuits["6-gate mixed"]; w = READ["bit 1"]
for lname, t in (("cubic", 0.0), ("exp", 0.0)):
    L = LOSS[lname](t)
    for S in (20, 40):
        seed += 10
        pred_inf = predicted(circ, th, w, L, S, method="param_shift_inf")
        b_inf, se_inf = measure(circ, th, w, L, S, seed=seed, method="param_shift_inf")
        b_fil, se_fil = measure(circ, th, w, L, S, seed=seed + 5, method="param_shift_filter")
        print(f"   {lname:5s} S={S:>3} _inf predicted [{fmt(pred_inf)}] measured [{fmt(b_inf)}] SE up to {se_inf.max():.4f}")
        print(f"   {lname:5s} S={S:>3} _filter measured [{fmt(b_fil)}] SE up to {se_fil.max():.4f}")

print("\n1d. k-batch remedy for a polynomial loss: cubic e^3 needs THREE independent batches (product E1 E2 E3), S=40, 20,000 repeats, param_shift_inf")
n6 = n_of(circ)
b_ = psc.BranchingSimulator(num_samples=40, diff_method="param_shift_inf")
Eh = lambda th_, k: b_.expval_all(b_.build_circuit(circ, [th_[i:i + 1] for i in range(n6)]), x_b, k)[1]
g1 = jax.jit(jax.vmap(jax.grad(lambda th_, k: Eh(th_, jax.random.fold_in(k, 0)) ** 3), in_axes=(None, 0)))
g2 = jax.jit(jax.vmap(jax.grad(lambda th_, k: Eh(th_, jax.random.fold_in(k, 0)) * Eh(th_, jax.random.fold_in(k, 1)) * Eh(th_, jax.random.fold_in(k, 0)) ), in_axes=(None, 0)))
g3 = jax.jit(jax.vmap(jax.grad(lambda th_, k: Eh(th_, jax.random.fold_in(k, 0)) * Eh(th_, jax.random.fold_in(k, 1)) * Eh(th_, jax.random.fold_in(k, 2))), in_axes=(None, 0)))
true3 = np.array(jax.grad(lambda th_: sv.expval_all(sv.build_circuit(circ, [th_[i:i + 1] for i in range(n6)]), x_sv)[1] ** 3)(th))
keys_ = jax.random.split(jax.random.key(901), 20000)
for name, gf in (("single batch (E1^3)", g1), ("two independent batches (E1 E2 E1)", g2), ("three independent batches (E1 E2 E3)", g3)):
    est_ = np.array(gf(th, keys_))
    print(f"   {name:38s} bias {fmt(est_.mean(0) - true3)}  (SE up to {(est_.std(0) / np.sqrt(len(est_))).max():.4f})")

# ---- 1e. param_shift_filter: delta-method formulas for a K=2 gate j (reps=1, no empty branches) ------------------------------
#   m_b, v_b = mean and variance of f with gate j forced to branch b;  Delta = m1 - m0;  p = sigmoid(theta_j);  g_j = p(1-p) Delta
#   bias_j ~ [ L2 p(1-p)(v1 - v0 - p Delta^2) + (1/2) L3 g_j Var f ] / S          (sign is gate dependent)
#   Var_j  ~ p(1-p) [ (1-p) v1 + p v0 + p^2 Delta^2 ] / S                            (linear readout)
def filter_pred(circ, th, w, L, S):
    E, V = E_and_var(circ, th, w)
    L2 = float(jax.grad(jax.grad(L))(E)); L3 = float(jax.grad(jax.grad(jax.grad(L)))(E))
    bias, var = [], []
    for j in range(n_of(circ)):
        pj = float(jax.nn.sigmoid(th[j]))
        m0, v0 = E_and_var(circ, th.at[j].set(-40.0), w); m1, v1 = E_and_var(circ, th.at[j].set(40.0), w)
        D = m1 - m0; g = pj * (1 - pj) * D
        bias.append((L2 * pj * (1 - pj) * (v1 - v0 - pj * D * D) + 0.5 * L3 * g * V) / S)
        var.append(pj * (1 - pj) * ((1 - pj) * v1 + pj * v0 + pj ** 2 * D * D) / S)
    return np.array(bias), np.sqrt(np.array(var))


circuits["3-gate chain"] = (psc.DiscretePCircuit([psc.PNOT(0), psc.PCNOT([0, 1]), psc.PNOT(1)]), jnp.array([0.3, -0.4, 0.8]))
circuits["2-gate"] = (psc.DiscretePCircuit([psc.PNOT(0), psc.PCNOT([0, 1])]), jnp.array([0.0, 0.0]))
circuits["depth-3 chain"] = (psc.DiscretePCircuit([psc.PNOT(0), psc.PCNOT([0, 1]), psc.PNOT(0)]), jnp.array([0.0, 0.0, 0.0]))
print("\n1e. param_shift_filter bias, delta-method prediction vs measured per gate (R as stated, independent keys)")
for cname, lname, t, S, R in (("6-gate mixed", "cubic", 0.0, 20, 20000), ("6-gate mixed", "cubic", 0.0, 40, 20000),
                              ("6-gate mixed", "exp", 0.0, 20, 20000), ("6-gate mixed", "exp", 0.0, 40, 20000),
                              ("3-gate chain", "quad", 0.8, 20, 20000), ("2-gate", "quad", 0.8, 20, 40000), ("2-gate", "quad", 0.8, 100, 200000)):
    circ, th = circuits[cname]; w = READ["bit 1"]; L = LOSS[lname](t)
    seed += 10
    pb, _sd = filter_pred(circ, th, w, L, S)
    mb, se = measure(circ, th, w, L, S, R=R, seed=seed, method="param_shift_filter")
    pq = predicted(circ, th, w, L, S)                      # _single, for the sign comparison
    print(f"   {cname:13s} {lname:5s} S={S:>3} _filter predicted [{fmt(pb)}]  measured [{fmt(mb)}]  SE up to {se.max():.4f}   (_single predicted [{fmt(pq)}])")
circ, th = circuits["depth-3 chain"]; w = READ["bit 1"]
_pb, psd = filter_pred(circ, th, w, LOSS["quad"](0.0), 1000)
bL = psc.BranchingSimulator(num_samples=1000, diff_method="param_shift_filter")
gL = jax.jit(jax.vmap(jax.grad(lambda th_, k: bL.expval_all(bL.build_circuit(circ, [th_[i:i + 1] for i in range(3)]), x_b, k)[1]), in_axes=(None, 0)))
msd = np.array(gL(th, jax.random.split(jax.random.key(4321), 2000))).std(0)
print(f"   depth-3 chain, linear readout, S=1000: _filter sd predicted [{fmt(psd)}] (mean {psd.mean():.5f})  measured [{fmt(msd)}] (mean {msd.mean():.5f})")

# ---- 1f. unified K-branch / reps form of the _filter bias (K=3 pdit gates; reps=2)  [form contributed by the reviewer, re-run here]
#   bias(theta_jk) = sum over reps r of [ L2 p_k (v_k - Var f) + (1/2) L3 g_{k,r} Var f ] / S,  g_{k,r} = p_k (m_k - E)
#   (m_k, v_k: mean and variance of f with gate j forced to branch k in rep r). For K=2 this equals the K=2 formula above.
def perm(mapping, n):
    M = np.zeros((n, n))
    for i, j in enumerate(mapping):
        M[j, i] = 1
    return M


def branch_probs(th):
    th = np.atleast_1d(np.asarray(th, dtype=float))
    if len(th) == 1:
        s_ = 1 / (1 + np.exp(-th[0])); return np.array([1 - s_, s_])
    z = np.concatenate([[0.0], th]); e = np.exp(z - z.max()); return e / e.sum()


def predict_k(branches, thetas, reps, f, x0, L2, L3, S):
    G = len(branches); seq = [(g, r) for r in range(reps) for g in range(G)]

    def run(force=None):
        rho = x0.copy()
        for (g, r) in seq:
            pk = branch_probs(thetas[g])
            K = branches[g][force[1]] if (force is not None and force[0] == (g, r)) else sum(q * B for q, B in zip(pk, branches[g]))
            rho = K @ rho
        return rho
    rho = run(); E = rho @ f; V = rho @ f ** 2 - E ** 2
    out = []
    for g in range(G):
        pk = branch_probs(thetas[g])
        for k in range(1, len(pk)):
            b_ = 0.0
            for r in range(reps):
                rk = run(((g, r), k)); mk = rk @ f; vk = rk @ f ** 2 - mk ** 2
                b_ += (L2 * pk[k] * (vk - V) + 0.5 * L3 * pk[k] * (mk - E) * V) / S
            out.append(b_)
    return np.array(out), E


def measure_list(circ, th_list, x_b_, x_sv_, fw, L, S, R=40000, seed=0):
    b = psc.BranchingSimulator(num_samples=S, diff_method="param_shift_filter")
    fwj = jnp.asarray(fw)
    flat = lambda gr: jnp.concatenate([jnp.ravel(a) for a in gr])
    lossb = lambda th, k: L(jnp.dot(fwj, b.expval_all(b.build_circuit(circ, th), x_b_, k)))
    gf = jax.jit(jax.vmap(lambda k: flat(jax.grad(lossb)(th_list, k))))
    G = np.concatenate([np.array(gf(jax.random.split(jax.random.key(seed + c), R // 4))) for c in range(4)])
    tr = np.array(flat(jax.grad(lambda th: L(jnp.dot(fwj, sv.expval_all(sv.build_circuit(circ, th), x_sv_))))(th_list)))
    return G.mean(0) - tr, G.std(0) / np.sqrt(len(G))


print("\n1f. unified K-branch / reps form of the _filter bias (40,000 repeats)")
cyc = [perm([0, 1, 2], 3), perm([1, 2, 0], 3), perm([2, 0, 1], 3)]
shf = [perm([0, 1, 2], 3), perm([1, 2, 0], 3)]
circA = psc.DiscretePCircuit([psc.PditCycle(0, dims=3), psc.PditShift(0, dims=3)])
thA = [np.array([0.3, -0.5]), np.array([0.2])]
fA = np.array([0.0, 1.0, 2.0]); xA = np.array([1.0, 0.0, 0.0])
for lname, S in (("quad", 20), ("quad", 50), ("cubic", 20), ("cubic", 50)):
    _, E0 = predict_k([cyc, shf], thA, 1, fA, xA, 0.0, 0.0, S)
    L = (lambda e: (e - 1.2) ** 2) if lname == "quad" else (lambda e: e ** 3)
    L2_, L3_ = (2.0, 0.0) if lname == "quad" else (6 * E0, 6.0)
    pred, _ = predict_k([cyc, shf], thA, 1, fA, xA, L2_, L3_, S)
    seed += 10
    bias, se = measure_list(circA, [jnp.array(t) for t in thA], jnp.zeros(1, jnp.int32), jnp.array(xA), np.array([1.0]), L, S, seed=seed)
    print(f"   K=3 PditCycle+PditShift {lname:5s} S={S:>3}  _filter predicted [{fmt(pred)}]  measured [{fmt(bias)}]  SE up to {se.max():.4f}")
I4 = np.eye(4)
pnot0 = [I4, perm([2, 3, 0, 1], 4)]
pcnot = [I4, perm([0, 1, 3, 2], 4)]
x4 = np.array([1.0, 0, 0, 0]); f1 = np.array([0.0, 1.0, 0.0, 1.0])
for S in (20, 50):
    circB = psc.DiscretePCircuit([psc.PNOT(0), psc.PCNOT([0, 1])], reps=2)
    thB = [np.array([0.4]), np.array([-0.3])]
    pred, _ = predict_k([pnot0, pcnot], thB, 2, f1, x4, 2.0, 0.0, S)
    seed += 10
    bias, se = measure_list(circB, [jnp.array(t) for t in thB], jnp.zeros(2, jnp.int32), jnp.array(x4), np.array([0.0, 1.0]), lambda e: (e - 0.8) ** 2, S, seed=seed)
    print(f"   reps=2 PNOT+PCNOT       quad  S={S:>3}  _filter predicted [{fmt(pred)}]  measured [{fmt(bias)}]  SE up to {se.max():.4f}")

# ---- 1g. empty branches: the delta-method formula assumes every branch is sampled. sampled.py computes
#   grad = p * (E1_hat - E_hat) * L'(E_hat), with E1_hat := 0 when branch 1 is empty (then E_hat = E0_hat, so it returns
#   -p E0_hat L'(E0_hat)) and E1_hat == E_hat when branch 0 is empty (it returns exactly 0). For a quadratic loss and a 0/1
#   observable the expected estimate is a finite sum over n1 ~ Binomial(S, p): within a branch the S samples are iid, so
#   E[E_b_hat] = m_b and E[E_b_hat^2] = m_b^2 + v_b / n_b. No expansion, so it covers the empty-branch events exactly.
from math import comb


def exact_quad(p, m0, m1, t, S):
    v0, v1 = m0 * (1 - m0), m1 * (1 - m1); tot = 0.0
    for n1 in range(S + 1):
        w = comb(S, n1) * p ** n1 * (1 - p) ** (S - n1); n0 = S - n1; q0, q1 = n0 / S, n1 / S
        if n1 == 0:
            val = -2 * p * (m0 ** 2 + v0 / S - t * m0)
        elif n1 == S:
            val = 0.0
        else:
            val = 2 * p * q0 * ((q0 - q1) * m0 * m1 + q1 * (m1 ** 2 + v1 / n1) - q0 * (m0 ** 2 + v0 / n0) - t * (m1 - m0))
        tot += w * val
    E = (1 - p) * m0 + p * m1
    return tot - p * (1 - p) * (m1 - m0) * 2 * (E - t)


print("\n1g. rare branches on gate 1, loss (E_hat - 0.8)^2 (40,000 repeats): delta method vs exact binomial sum vs measured")
Lq = lambda e: (e - 0.8) ** 2
pnot1 = [np.eye(4), perm([1, 0, 3, 2], 4)]
cases = (("2-gate", [psc.PNOT(0), psc.PCNOT([0, 1])], [pnot0, pcnot], [0.0], -3.0, 20),
         ("2-gate", [psc.PNOT(0), psc.PCNOT([0, 1])], [pnot0, pcnot], [0.0], -3.0, 100),
         ("2-gate", [psc.PNOT(0), psc.PCNOT([0, 1])], [pnot0, pcnot], [0.0], -2.0, 20),
         ("3-gate", [psc.PNOT(0), psc.PCNOT([0, 1]), psc.PNOT(1)], [pnot0, pcnot, pnot1], [0.0, 0.5], -3.0, 20),
         ("3-gate", [psc.PNOT(0), psc.PCNOT([0, 1]), psc.PNOT(1)], [pnot0, pcnot, pnot1], [0.0, 0.5], 3.0, 20),
         ("3-gate", [psc.PNOT(0), psc.PCNOT([0, 1]), psc.PNOT(1)], [pnot0, pcnot, pnot1], [0.0, 0.5], -2.0, 20))
for name, gates, mats, rest, th0, S in cases:
    thC = [np.array([th0])] + [np.array([v]) for v in rest]
    pred, E = predict_k(mats, thC, 1, f1, x4, 2.0, 0.0, S)
    pp = 1 / (1 + np.exp(-th0))

    def m_forced(bb):
        rho = mats[0][bb] @ x4
        for B, tv in zip(mats[1:], rest):
            q = 1 / (1 + np.exp(-tv)); rho = ((1 - q) * B[0] + q * B[1]) @ rho
        return rho @ f1
    m0, m1 = m_forced(0), m_forced(1)
    seed += 10
    bias, se = measure_list(psc.DiscretePCircuit(gates), [jnp.array(t) for t in thC], jnp.zeros(2, jnp.int32), jnp.array(x4),
                            np.array([0.0, 1.0]), Lq, S, seed=seed)
    print(f"   {name} m0={m0:.3f} m1={m1:.3f} theta0={th0:+.0f} S={S:>3} P(n1=0)={(1 - pp) ** S:.3f} P(n0=0)={pp ** S:.3f}: "
          f"delta method {pred[0]:.5f}, exact {exact_quad(pp, m0, m1, 0.8, S):.5f}, measured {bias[0]:.5f} (SE {se[0]:.5f})")

# ---- 2. fixed points ------------------------------------------------------------------------------------------------------------
c1 = psc.DiscretePCircuit([psc.PNOT(0)])
x1 = jnp.zeros(1, jnp.int32)
t = 0.8


def tail_E(method, S, seed, lr=0.5, steps=3000, tail=1000):
    b = psc.BranchingSimulator(num_samples=S, diff_method=method)
    g = jax.grad(lambda th, k: (b.expval_all(b.build_circuit(c1, [th]), x1, k)[0] - t) ** 2)

    def step(th, k):
        th = th - lr * g(th, k)
        return th, jax.nn.sigmoid(th[0])
    _, Es = jax.lax.scan(step, jnp.array([0.0]), jax.random.split(jax.random.key(seed), steps))
    return float(np.mean(np.array(Es)[-tail:]))


print("\n2. single PNOT, loss (E_hat - 0.8)^2, plain SGD lr 0.5, 3000 steps, tail-averaged exact E (mean of 4 seeds): fixed point vs closed form")
print(f"   {'S':>3} {'method':10s} {'measured':>9} {'predicted':>10}")
for S in (20, 50):
    for m, pred in (("param_shift_inf", t), ("param_shift_single", t - 1 / S), ("param_shift_filter", t / (1 - 1 / S))):
        val = np.mean([tail_E(m, S, sd) for sd in range(4)])
        print(f"   {S:>3} {m[12:]:10s} {val:>9.4f} {pred:>10.4f}")

# 3-gate chain: deterministic Adam (same settings as cookbook_12 part 1b) on the EXPECTED biased gradient = true gradient + 2 Var(f)/S per gate
c3 = psc.DiscretePCircuit([psc.PNOT(0), psc.PCNOT([0, 1]), psc.PNOT(1)])
exact_E = lambda th: sv.expval_all(sv.build_circuit(c3, [th[0:1], th[1:2], th[2:3]]), x_sv)[1]
S = 20


def expected_biased_grad(th):
    E = exact_E(th)
    return jax.grad(lambda z: (exact_E(z) - t) ** 2)(th) + 2.0 * E * (1.0 - E) / S


def run_det(steps):
    opt = optax.adam(0.05); th = jnp.array([0.3, -0.4, 0.8]); st = opt.init(th)
    step = jax.jit(lambda th, st: (lambda u: (optax.apply_updates(th, u[0]), u[1]))(opt.update(expected_biased_grad(th), st)))
    for _ in range(steps):
        th, st = step(th, st)
    return th


for steps in (400, 4000):
    th_ = run_det(steps)
    print(f"\n   3-gate chain, S={S}: deterministic Adam on [true gradient + 2 Var(f)/S], {steps} steps: E = {float(exact_E(th_)):.4f}, theta = [{fmt(th_)}]")
print("   The constant bias pushes the first two logits to -infinity (their dE/dtheta vanishes, nothing balances it); the last gate then satisfies "
      "the single-gate condition E* = t - 1/S. So 0.7500 is a degenerate limit of the coupled gates, not a joint fixed point; the sampled _single "
      "single-batch runs (cookbook_12 part 1b, 6 seeds) stop short of it: 0.7367 +/- 0.0313")

# ---- 3. Eq. 44 ------------------------------------------------------------------------------------------------------------------------
# toy gate: G(theta)(y|x) = exp(-E(x,y)) / Z(x), E(x, y) = -(a + b x)(2y - 1); x ~ Bernoulli(0.4); readout f = y
pxx = 0.4
theta = jnp.array([0.3, -0.7])
Eng = lambda th, x, y: -(th[0] + th[1] * x) * (2 * y - 1)
dE = jax.grad(Eng, argnums=0)


def J_exact(th):
    return sum(px * jax.nn.sigmoid(2 * (th[0] + th[1] * x)) for x, px in ((0, 1 - pxx), (1, pxx)))


g_exact = np.array(jax.grad(J_exact)(theta))


def score_sample(key):
    kx, ky = jax.random.split(key)
    x = jax.random.bernoulli(kx, pxx).astype(jnp.float32)
    p1 = jax.nn.sigmoid(2 * (theta[0] + theta[1] * x))
    y = jax.random.bernoulli(ky, p1).astype(jnp.float32)
    inner = p1 * dE(theta, x, 1.0) + (1 - p1) * dE(theta, x, 0.0)
    return y * (-dE(theta, x, y) + inner)


def score_sample_aux(key):
    kx, ky, ka = jax.random.split(key, 3)
    x = jax.random.bernoulli(kx, pxx).astype(jnp.float32)
    p1 = jax.nn.sigmoid(2 * (theta[0] + theta[1] * x))
    y = jax.random.bernoulli(ky, p1).astype(jnp.float32)
    y2 = jax.random.bernoulli(ka, p1).astype(jnp.float32)
    return y * (-dE(theta, x, y) + dE(theta, x, y2))


keys = jax.random.split(jax.random.key(7), 400000)
est = np.array(jax.vmap(score_sample)(keys)); est_aux = np.array(jax.vmap(score_sample_aux)(keys))
print(f"\n3. paper Eq. 44 on a Boltzmann conditional gate, exact gradient {g_exact.round(4)}")
print(f"   score term, inner expectation exact : {est.mean(0).round(4)} +/- {(est.std(0) / np.sqrt(len(est))).round(4)}")
print(f"   score term, one auxiliary sample y' : {est_aux.mean(0).round(4)} +/- {(est_aux.std(0) / np.sqrt(len(est_aux))).round(4)} "
      f"(sd per sample [{est_aux.std(0)[0]:.3f} {est_aux.std(0)[1]:.3f}] vs [{est.std(0)[0]:.3f} {est.std(0)[1]:.3f}]: higher variance, as the paper warns)")

# 3b: the energy-based gate feeds a later gate: z = y flipped with probability q; readout f = z. Eq. 44 weights the gate's score by f(z_k).
q = 0.2
def J2_exact(th):
    return sum(px * (jax.nn.sigmoid(2 * (th[0] + th[1] * x)) * (1 - q) + (1 - jax.nn.sigmoid(2 * (th[0] + th[1] * x))) * q)
               for x, px in ((0, 1 - pxx), (1, pxx)))
g2_exact = np.array(jax.grad(J2_exact)(theta))


def chain_sample(key, mode):
    kx, ky, ka, kf = jax.random.split(key, 4)
    x = jax.random.bernoulli(kx, pxx).astype(jnp.float32)
    p1 = jax.nn.sigmoid(2 * (theta[0] + theta[1] * x))
    y = jax.random.bernoulli(ky, p1).astype(jnp.float32)
    z = jnp.abs(y - jax.random.bernoulli(kf, q).astype(jnp.float32))           # z = y XOR flip
    if mode == "exact":
        score = -dE(theta, x, y) + p1 * dE(theta, x, 1.0) + (1 - p1) * dE(theta, x, 0.0)
    else:
        y2 = jax.random.bernoulli(ka, p1).astype(jnp.float32)
        score = -dE(theta, x, y) + dE(theta, x, y2)
    return z, score


zs, sc_exact = jax.vmap(lambda k: chain_sample(k, "exact"))(keys)
zs2, sc_aux = jax.vmap(lambda k: chain_sample(k, "aux"))(keys)
zs, sc_exact, zs2, sc_aux = map(np.array, (zs, sc_exact, zs2, sc_aux))
rows = {"inner expectation exact (= REINFORCE, exact score)": zs[:, None] * sc_exact,
        "one auxiliary sample y'": zs2[:, None] * sc_aux,
        "auxiliary sample + baseline (f - mean f)": (zs2 - zs2.mean())[:, None] * sc_aux,
        "exact score + baseline (f - mean f)": (zs - zs.mean())[:, None] * sc_exact}
print(f"\n3b. energy-based gate feeding a later gate (flip probability {q}), readout f = z, exact gradient {g2_exact.round(4)}")
for name, e in rows.items():
    print(f"   {name:52s} mean {e.mean(0).round(4)} +/- {(e.std(0) / np.sqrt(len(e))).round(4)}  sd per sample [{e.std(0)[0]:.3f} {e.std(0)[1]:.3f}]")


# 3c: energy-based gate with FIXED input (so it is a sigmoid-mixture gate and parameter shift applies), then two downstream gates.
# E(y) = -a (2y - 1): P(y=1) = sigmoid(2a) = sigmoid(theta1), theta1 = 2a. Downstream PNOT-like flips with logits th2, th3. Readout: final bit.
# Compare, at EQUAL circuit runs T = 7000 per gradient, d/da from: Eq. 44 (exact score; aux sample; each with baseline) and Torx param_shift_inf (x 2).
a0 = 0.2
th_chain = jnp.array([2 * a0, 0.5, -0.3])
c_chain = psc.DiscretePCircuit([psc.PNOT(0), psc.PNOT(0), psc.PNOT(0)])
x_c = jnp.zeros(1, jnp.int32); x_cs = jnp.array([1.0, 0.0])
Ec = lambda th_: sv.expval_all(sv.build_circuit(c_chain, [th_[0:1], th_[1:2], th_[2:3]]), x_cs)[0]
true_da = 2 * float(jax.grad(Ec)(th_chain)[0])
print(f"\n3c. energy-based gate (fixed input) followed by two flip gates, readout = final bit; exact dJ/da = {true_da:.4f}; equal budget T = 7000 circuit runs per gradient, 200 repeats")
T_ = 7000
bps = psc.BranchingSimulator(num_samples=T_ // 7, diff_method="param_shift_inf")
gps = jax.jit(jax.grad(lambda th_, k: bps.expval_all(bps.build_circuit(c_chain, [th_[0:1], th_[1:2], th_[2:3]]), x_c, k)[0]))
ps = np.array([2 * float(gps(th_chain, jax.random.key(3000 + i))[0]) for i in range(200)])


def eq44_batch(key, mode):
    ky, k2, k3, ka = jax.random.split(key, 4)
    p1 = jax.nn.sigmoid(th_chain[0])
    y = jax.random.bernoulli(ky, p1, (T_,)).astype(jnp.float32)
    f2 = jax.random.bernoulli(k2, jax.nn.sigmoid(th_chain[1]), (T_,)).astype(jnp.float32)
    f3 = jax.random.bernoulli(k3, jax.nn.sigmoid(th_chain[2]), (T_,)).astype(jnp.float32)
    z = jnp.abs(jnp.abs(y - f2) - f3)                                       # y flipped twice
    if mode == "exact":
        score = 2.0 * (y - p1)                                              # -dE/da + E_y' dE/da with E(y) = -a(2y-1)
    else:
        y2 = jax.random.bernoulli(ka, p1, (T_,)).astype(jnp.float32)
        score = 2.0 * (y - y2)
    return jnp.mean(z * score), jnp.mean((z - z.mean()) * score)


eq_exact = np.array([[float(v) for v in eq44_batch(jax.random.key(6000 + i), "exact")] for i in range(200)])
eq_aux = np.array([[float(v) for v in eq44_batch(jax.random.key(7000 + i), "aux")] for i in range(200)])
print(f"   parameter shift (_inf, S={T_ // 7}, x2)        mean {ps.mean():.4f}  sd {ps.std(ddof=1):.4f}   (buys all 3 gradients)")
# d/da only: two deterministic branch circuits for the first gate, T/2 samples each:  dJ/da = 2 p(1-p) (E1 - E0)
bdet = psc.BranchingSimulator(num_samples=T_ // 2)
p_a = float(jax.nn.sigmoid(th_chain[0]))
def two_branch(key):
    k0, k1 = jax.random.split(key)
    E0 = bdet.expval_all(bdet.build_circuit(c_chain, [jnp.array([-40.0]), th_chain[1:2], th_chain[2:3]]), x_c, k0)[0]
    E1 = bdet.expval_all(bdet.build_circuit(c_chain, [jnp.array([40.0]), th_chain[1:2], th_chain[2:3]]), x_c, k1)[0]
    return 2 * p_a * (1 - p_a) * (E1 - E0)
tb = np.array([float(two_branch(jax.random.key(8000 + i))) for i in range(200)])
m0_ = float(Ec(jnp.array([-40.0, th_chain[1], th_chain[2]]))); m1_ = float(Ec(jnp.array([40.0, th_chain[1], th_chain[2]])))
pred_sd = 2 * p_a * (1 - p_a) * np.sqrt((m0_ * (1 - m0_) + m1_ * (1 - m1_)) / (T_ // 2))
print(f"   parameter shift for d/da only (2 x {T_ // 2})  mean {tb.mean():.4f}  sd {tb.std(ddof=1):.4f}   (formula {pred_sd:.4f})")
print(f"   Eq. 44 exact score, no baseline          mean {eq_exact[:, 0].mean():.4f}  sd {eq_exact[:, 0].std(ddof=1):.4f}")
print(f"   Eq. 44 exact score, baseline f - mean f  mean {eq_exact[:, 1].mean():.4f}  sd {eq_exact[:, 1].std(ddof=1):.4f}")
print(f"   Eq. 44 aux sample, no baseline           mean {eq_aux[:, 0].mean():.4f}  sd {eq_aux[:, 0].std(ddof=1):.4f}")
print(f"   Eq. 44 aux sample, baseline              mean {eq_aux[:, 1].mean():.4f}  sd {eq_aux[:, 1].std(ddof=1):.4f}")


# full 3-parameter gradient at equal cost T=7000: _inf (S=1000), _filter (S=7000), REINFORCE with a leave-one-out baseline (M=7000)
bf = psc.BranchingSimulator(num_samples=T_, diff_method="param_shift_filter")
gf_ = jax.jit(jax.grad(lambda th_, k: bf.expval_all(bf.build_circuit(c_chain, [th_[0:1], th_[1:2], th_[2:3]]), x_c, k)[0]))
gi_ = jax.jit(jax.grad(lambda th_, k: bps.expval_all(bps.build_circuit(c_chain, [th_[0:1], th_[1:2], th_[2:3]]), x_c, k)[0]))
true_full = np.array(jax.grad(Ec)(th_chain))


def reinforce_full(key):
    ks = jax.random.split(key, 3)
    p_ = jax.nn.sigmoid(th_chain)
    bb = jnp.stack([jax.random.bernoulli(ks[i], p_[i], (T_,)) for i in range(3)], 1).astype(jnp.float32)
    z = jnp.mod(bb.sum(1), 2.0)                                        # three flips of an initial 0
    base = (z.sum() - z) / (T_ - 1)
    return jnp.mean((z - base)[:, None] * (bb - p_[None, :]), 0)        # d log p / d theta = b - p for a logit gate


rf = np.array([np.array(jax.jit(reinforce_full)(jax.random.key(9000 + i))) for i in range(200)])
fi = np.array([np.array(gf_(th_chain, jax.random.key(9500 + i))) for i in range(200)])
ii = np.array([np.array(gi_(th_chain, jax.random.key(9800 + i))) for i in range(200)])
print(f"\n   full gradient (all 3 logits) at T = {T_} runs, 200 repeats; exact {fmt(true_full)}")
for name, arr in (("_inf (S=1000)", ii), ("_filter (S=7000)", fi), ("REINFORCE + leave-one-out baseline (M=7000)", rf)):
    print(f"   {name:44s} mean [{fmt(arr.mean(0))}]  sd [{fmt(arr.std(0, ddof=1))}]")
