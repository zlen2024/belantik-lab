"""Cookbook 11: a training loop that is actually run, the Jensen trap, REINFORCE baselines, the discrete-gate
score term, gate truth tables from a run, Mix / GaussianNoiseGate, StateVectorSimulator scaling, vmap batching.
Run from OUTSIDE master/.
"""
import time
import jax, jax.numpy as jnp, numpy as np
import optax
from torx import psc

sv = psc.StateVectorSimulator()
sig = jax.nn.sigmoid
c3 = psc.DiscretePCircuit([psc.PNOT(0), psc.PCNOT([0, 1]), psc.PNOT(1)])
x_sv, x_b = jnp.zeros(4).at[0].set(1.0), jnp.zeros(2, jnp.int32)


def exact_E(th):                                           # E[bit 1], exact
    return sv.expval_all(sv.build_circuit(c3, [th[0:1], th[1:2], th[2:3]]), x_sv)[1]

# ---- 1. training loop: maximise E[bit 1] with Adam, four gradient sources ---------------------------------
def sampled_E(method, S):
    b = psc.BranchingSimulator(num_samples=S, diff_method=method)
    return lambda th, k: b.expval_all(b.build_circuit(c3, [th[0:1], th[1:2], th[2:3]]), x_b, k)[1]

def run_train(grad_fn, steps=300, lr=0.1, seed=0):
    th = jnp.array([0.3, -0.4, 0.8]); opt = optax.adam(lr); st = opt.init(th)
    key = jax.random.key(seed)
    @jax.jit
    def step(th, st, k):
        g = -grad_fn(th, k)                                # minimise -E
        upd, st = opt.update(g, st); return optax.apply_updates(th, upd), st
    for i in range(steps):
        key, k = jax.random.split(key); th, st = step(th, st, k)
    return th

def reinforce_grad(th, key, M=2000):
    def run(theta, kk):
        k = jax.random.split(kk, 3); p = sig(theta)
        bb = jax.vmap(lambda a, pp: jax.random.bernoulli(a, pp))(k, p).astype(jnp.int32)
        z1 = (bb[0] & bb[1]) ^ bb[2]
        return z1.astype(jnp.float32), jnp.sum(jnp.where(bb == 1, jnp.log(sig(theta)), jnp.log1p(-sig(theta))))
    def surrogate(theta):
        f, lp = jax.vmap(lambda kk: run(theta, kk))(jax.random.split(key, M))
        return jnp.mean(jax.lax.stop_gradient(f - f.mean()) * lp) * M / (M - 1)
    return jax.grad(surrogate)(th)

cands = {"exact (state vector)": lambda th, k: jax.grad(exact_E)(th),
         "param_shift_inf S=500": lambda th, k: jax.grad(sampled_E("param_shift_inf", 500))(th, k),
         "param_shift_filter S=3000": lambda th, k: jax.grad(sampled_E("param_shift_filter", 3000))(th, k),
         "REINFORCE M=3000": lambda th, k: reinforce_grad(th, k, 3000)}
print("1. maximise E[bit 1] (max possible 1.0), Adam lr 0.1, 300 steps; final EXACT objective:")
for name, fn in cands.items():
    th = run_train(fn)
    print(f"   {name:28s} E = {float(exact_E(th)):.4f}   theta = {np.array(th).round(2)}")

# ---- 2. the Jensen trap: a loss nonlinear in the sampled expectation can have a biased gradient --------------------
tgt, R = 0.8, 3000
th0 = jnp.array([0.3, -0.4, 0.8])
true_g = np.array(jax.grad(lambda t: (exact_E(t) - tgt) ** 2)(th0))
print("\n2. bias of the gradient of (E_hat - %s)^2 vs S (%d repeats; true gradient %s); bias = mean - true, SE in brackets" % (tgt, R, true_g.round(4)))
ks = jax.random.split(jax.random.key(1), (R, 2))
for S in (20, 100, 1000):
    for m in ("param_shift_inf", "param_shift_single", "param_shift_filter"):
        b = psc.BranchingSimulator(num_samples=S, diff_method=m)
        Ehat = lambda th, k: b.expval_all(b.build_circuit(c3, [th[0:1], th[1:2], th[2:3]]), x_b, k)[1]
        single = jax.jit(jax.grad(lambda th, k: (Ehat(th, k) - tgt) ** 2))
        prod = jax.jit(jax.grad(lambda th, k1, k2: (Ehat(th, k1) - tgt) * (Ehat(th, k2) - tgt)))
        gs = np.array([np.array(single(th0, k[0])) for k in ks])
        gp = np.array([np.array(prod(th0, k[0], k[1])) for k in ks])
        print(f"   S={S:>4} {m:20s} single batch bias {(gs.mean(0) - true_g).round(4)} (SE {(gs.std(0) / np.sqrt(R)).round(4)});"
              f"  two-batch product bias {(gp.mean(0) - true_g).round(4)} (SE {(gp.std(0) / np.sqrt(R)).round(4)})")

Ex0 = float(exact_E(th0))
print("   law for the 3-gate chain: E = %.4f, 2 E(1-E) = %.3f; predicted bias 2E(1-E)/S: %s" % (Ex0, 2 * Ex0 * (1 - Ex0), ", ".join("S=%d: %.5f" % (S_, 2 * Ex0 * (1 - Ex0) / S_) for S_ in (20, 100, 1000))))
b20 = psc.BranchingSimulator(num_samples=20, diff_method="param_shift_single")
g20 = jax.jit(jax.grad(lambda th, k: (b20.expval_all(b20.build_circuit(c3, [th[0:1], th[1:2], th[2:3]]), x_b, k)[1] - 0.8) ** 2))
m20 = np.mean([np.array(g20(th0, jax.random.key(500 + i))) for i in range(3000)], 0)
print("   _single at S=20: mean gradient %s vs true %s (ratio %s)" % (m20.round(3), true_g.round(4), (m20 / true_g).round(1)))

# ---- 2b. exact law: bias of param_shift_single on (E_hat - t)^2 is 2 Var(f) / S (same on every gate) -------------------
# derivation: grad = grad_out * (E_hat - E_tilde'), grad_out = 2 (E_hat - t) is built from the SAME primal samples as E_hat,
#             so the bias is 2 Cov(E_hat, E_hat) = 2 Var(E_hat) = 2 Var(f) / S.  For a bit readout Var(f) = E (1 - E) <= 1/4.
print("\n2b. law check, bias(_single) = 2 E(1-E)/S; two circuits, loss (E_hat - t)^2:")
c4 = psc.DiscretePCircuit([psc.PNOT(0), psc.PCNOT([0, 1]), psc.PNOT(0), psc.PNOT(1)])
for name, circ, thv, t_, xsv_, xb_, f_idx in (("3-gate chain, t=0.8", c3, th0, 0.8, x_sv, x_b, 1),
                                           ("4-gate circuit, t=0.3", c4, jnp.array([0.3, -0.4, 0.5, 0.8]), 0.3, x_sv, x_b, 1)):
    n_g = len(circ.gates)
    th_list = lambda th: [th[i:i + 1] for i in range(n_g)]
    Ex = float(sv.expval_all(sv.build_circuit(circ, th_list(thv)), xsv_)[f_idx])
    true_gg = np.array(jax.grad(lambda th: (sv.expval_all(sv.build_circuit(circ, th_list(th)), xsv_)[f_idx] - t_) ** 2)(thv))
    for S in (40, 160):
        b = psc.BranchingSimulator(num_samples=S, diff_method="param_shift_single")
        g = jax.jit(jax.grad(lambda th, k: (b.expval_all(b.build_circuit(circ, th_list(th)), xb_, k)[f_idx] - t_) ** 2))
        gm = np.array([np.array(g(thv, jax.random.key(i))) for i in range(4000)])
        print(f"   {name}: E={Ex:.3f}, S={S}: predicted {2 * Ex * (1 - Ex) / S:.4f}, measured bias per gate {(gm.mean(0) - true_gg).round(4)}")

# ---- 3. REINFORCE baselines: batch mean (needs M/(M-1)) vs leave-one-out ---------------------------------------
def make(kind):
    def f(th, key, M=1000):
        k = jax.random.split(key, M)
        def one(kk):
            kk3 = jax.random.split(kk, 3); p = sig(th)
            bb = jax.vmap(lambda a, pp: jax.random.bernoulli(a, pp))(kk3, p).astype(jnp.int32)
            return ((bb[0] & bb[1]) ^ bb[2]).astype(jnp.float32), jnp.sum(jnp.where(bb == 1, jnp.log(p), jnp.log1p(-p)))
        ff, lp = jax.vmap(one)(k)
        if kind == "loo":
            base = (ff.sum() - ff) / (M - 1)
            return jnp.mean(jax.lax.stop_gradient(ff - base) * lp)
        if kind == "none":
            return jnp.mean(jax.lax.stop_gradient(ff) * lp)
        return jnp.mean(jax.lax.stop_gradient(ff - ff.mean()) * lp) * M / (M - 1)
    return jax.jit(jax.grad(f))
ex = np.array(jax.grad(exact_E)(th0))
print("\n3. REINFORCE baselines, M=1000, 200 repeats, theta=[0.3,-0.4,0.8]:")
for kind in ("none", "batch-mean x M/(M-1)", "loo"):
    g = make({"none": "none", "batch-mean x M/(M-1)": "bm", "loo": "loo"}[kind])
    est = np.array([np.array(g(th0, jax.random.key(i))) for i in range(200)])
    print(f"   {kind:22s} sd {est.std(0).mean():.4f}   max|bias| {np.abs(est.mean(0) - ex).max():.4f}")

# ---- 4. discrete gate inside a hybrid circuit: pathwise gradient misses the score term -------------------------
means = jnp.array([[2.0], [0.0], [-2.0]])
hc = psc.HybridPCircuit([psc.PditCycle(0, dims=3), psc.MixtureGaussianGate(sites=(0, 0), dims=(1,), num_components=3)])
th_d = jnp.array([0.5, -0.5])
def Ex(theta):                                              # analytic E[x]: state = stay 0 / forward 1 / backward 2
    return jnp.sum(jax.nn.softmax(jnp.concatenate([jnp.zeros(1), theta])) * means[:, 0])
true_gx = np.array(jax.grad(Ex)(th_d))
state0 = {"discrete": jnp.array([0], jnp.int32), "continuous": jnp.zeros(1)}
params = lambda t: [t, {"means": means, "log_vars": jnp.full((3, 1), -2.0)}]
path = jax.jit(jax.vmap(jax.grad(lambda t, k: hc.sample(k, state0, params(t))["continuous"][0]), in_axes=(None, 0)))
ksx = jax.random.split(jax.random.key(3), 20000)
pw = np.array(path(th_d, ksx)).mean(0)
def score_term(t, k):
    kb, kn = jax.random.split(k)
    lp_all = jax.nn.log_softmax(jnp.concatenate([jnp.zeros(1), t]))
    br = jax.random.categorical(kb, lp_all)
    x = means[br, 0] + jnp.exp(-1.0) * jax.random.normal(kn)
    return x, lp_all[br]
def sc(t, k):
    x, lp = score_term(t, k); return jax.lax.stop_gradient(x) * lp
sgs = np.array(jax.vmap(jax.grad(sc), in_axes=(None, 0))(th_d, jax.random.split(jax.random.key(4), 400000)))
sg, sg_se = sgs.mean(0), sgs.std(0) / np.sqrt(len(sgs))
print(f"\n4. E[x] gradient wrt the PditCycle theta: analytic {true_gx.round(4)}, pathwise through sample() {pw.round(4)} "
      f"(zero: biased), score-function term {sg.round(4)} +/- {sg_se.round(4)} (SE)")

# ---- 5. truth tables from a run ------------------------------------------------------------------------------------
def table(gate, n):
    c = psc.DiscretePCircuit([gate]); comp = sv.build_circuit(c, [jnp.array([20.0])]); out = []
    for i in range(2 ** n):
        x = jnp.zeros(2 ** n).at[i].set(1.0); d = np.array(sv.density(comp, x))
        out.append(f"{i:0{n}b}->{int(d.argmax()):0{n}b}")
    return " ".join(out)
print("\n5. truth tables from a run (theta=+20, site 0 = most significant bit):")
print("   PMultiCNOT([0,1,2]):", table(psc.PMultiCNOT([0, 1, 2]), 3))
print("   PCSWAP([0,1,2])    :", table(psc.PCSWAP([0, 1, 2]), 3))
print("   PDEMUX([0,1])      :", table(psc.PDEMUX([0, 1]), 2))

# ---- 6. Mix and GaussianNoiseGate -------------------------------------------------------------------------------------
c_mix = psc.HybridPCircuit([psc.Mix(sites=[0, 1], dims=(1, 1))])
o = c_mix.sample(jax.random.key(0), {"discrete": jnp.zeros((0,), jnp.int32), "continuous": jnp.array([1.0, 0.0])}, [jnp.array(jnp.pi / 2)])
c_n = psc.HybridPCircuit([psc.GaussianNoiseGate(sites=[0], dims=(1,))])
xs = c_n.sample_multiple(jax.random.key(1), {"discrete": jnp.zeros((0,), jnp.int32), "continuous": jnp.zeros(1)},
                         [jnp.array([np.log(0.25)])], n_samples=20000)["continuous"][:, 0]
print(f"\n6. Mix(theta=pi/2) rotates (1,0) -> {np.array(o['continuous']).round(3)};  GaussianNoiseGate(log_var=log 0.25) var {float(xs.var()):.3f} (0.25)")

# ---- 7. StateVectorSimulator scaling: 2^n probability vector, n gates ---------------------------------------------------
# jax.jit(sv.density) FAILS (compiled-circuit dims are not static); use equinox.filter_jit. Un-jitted calls are dominated by dispatch.
import equinox as eqx
try:
    jax.jit(sv.density)(sv.build_circuit(psc.DiscretePCircuit([psc.PNOT(i) for i in range(4)]), [jnp.array([0.1])] * 4), jnp.zeros(16).at[0].set(1.0))
except TypeError as e:
    print("\n7. jax.jit(sv.density) ->", type(e).__name__ + ":", str(e)[:70], "...")
print("   StateVectorSimulator.density wall time (n PNOT gates), eqx.filter_jit vs un-jitted:")
fj = eqx.filter_jit(sv.density)
for n in (10, 16, 20, 22, 24):
    c = psc.DiscretePCircuit([psc.PNOT(i) for i in range(n)])
    comp = sv.build_circuit(c, [jnp.array([0.1])] * n); x = jnp.zeros(2 ** n).at[0].set(1.0)
    fj(comp, x).block_until_ready()
    t0 = time.time(); fj(comp, x).block_until_ready(); dt = time.time() - t0
    t1 = time.time(); sv.density(comp, x).block_until_ready(); du = time.time() - t1
    print(f"   n={n:>2}: 2^n = {2 ** n:>9} states, jitted {dt * 1000:8.1f} ms, un-jitted {du * 1000:7.0f} ms")

# ---- 8. batching many circuits (thetas) with vmap ----------------------------------------------------------------------
bb = psc.BranchingSimulator(num_samples=500, diff_method="param_shift_inf")
f = lambda th, k: bb.expval_all(bb.build_circuit(c3, [th[0:1], th[1:2], th[2:3]]), x_b, k)[1]
thetas = jnp.stack([th0 + 0.1 * i for i in range(8)])
vals = jax.jit(jax.vmap(f))(thetas, jax.random.split(jax.random.key(5), 8))
grads_b = jax.jit(jax.vmap(jax.grad(f)))(thetas, jax.random.split(jax.random.key(6), 8))
print(f"\n8. vmap over 8 parameter sets: values {np.array(vals).round(3)}, gradient batch shape {grads_b.shape}")
