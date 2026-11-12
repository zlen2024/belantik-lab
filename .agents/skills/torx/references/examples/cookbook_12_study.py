"""Cookbook 12: training study at equal cost, failure cases with real messages, condition() hand arithmetic,
DFG nesting, PditShift/PditSWAP tables, and a Torx-vs-THRML equivalence run. Run from OUTSIDE master/.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax, jax.numpy as jnp, numpy as np
import optax
import equinox as eqx
from torx import AbstractReferenceFactor, DFG, Site, psc

sv = psc.StateVectorSimulator()
sig = jax.nn.sigmoid
c3 = psc.DiscretePCircuit([psc.PNOT(0), psc.PCNOT([0, 1]), psc.PNOT(1)])
x_sv, x_b = jnp.zeros(4).at[0].set(1.0), jnp.zeros(2, jnp.int32)
exact_E = lambda th: sv.expval_all(sv.build_circuit(c3, [th[0:1], th[1:2], th[2:3]]), x_sv)[1]

# ---- 1. training at equal cost: T = 7000 circuit runs per gradient ------------------------------------------------
T = 7000
def sampled(method, S):
    b = psc.BranchingSimulator(num_samples=S, diff_method=method)
    return jax.grad(lambda th, k: b.expval_all(b.build_circuit(c3, [th[0:1], th[1:2], th[2:3]]), x_b, k)[1])

def reinforce(th, key, M=T):
    def one(t, kk):                                        # t is the DIFFERENTIATED parameter (closing over th gives zero gradient)
        k = jax.random.split(kk, 3); p = sig(t)
        bb = jax.vmap(lambda a, pp: jax.random.bernoulli(a, pp))(k, p).astype(jnp.int32)
        return ((bb[0] & bb[1]) ^ bb[2]).astype(jnp.float32), jnp.sum(jnp.where(bb == 1, jnp.log(p), jnp.log1p(-p)))
    def sur(t):
        f, lp = jax.vmap(lambda kk: one(t, kk))(jax.random.split(key, M))
        base = (f.sum() - f) / (M - 1)
        return jnp.mean(jax.lax.stop_gradient(f - base) * lp)
    return jax.grad(sur)(th)

# runs per gradient: inf (1+G*K)S = 7S, single (1+G)S = 4S, filter S, REINFORCE M
methods = {"param_shift_inf (S=1000)": sampled("param_shift_inf", T // 7),
           "param_shift_single (S=1750)": sampled("param_shift_single", T // 4),
           "param_shift_filter (S=7000)": sampled("param_shift_filter", T),
           "REINFORCE (M=7000)": reinforce}

def train(grad_fn, lr, steps=150, seed=0):
    th = jnp.array([0.3, -0.4, 0.8]); opt = optax.adam(lr); st = opt.init(th); key = jax.random.key(seed)
    @jax.jit
    def step(th, st, k):
        g = -grad_fn(th, k); u, st = opt.update(g, st); return optax.apply_updates(th, u), st, exact_E(th)
    reach = None
    for i in range(steps):
        key, k = jax.random.split(key); th, st, e = step(th, st, k)
        if reach is None and float(e) >= 0.95: reach = i
    return float(exact_E(th)), reach

SEEDS = 6
print(f"1. maximise E[bit 1] at equal cost ({T} circuit runs per gradient), Adam, 150 steps, {SEEDS} seeds: final EXACT objective, steps to reach 0.95")
for lr in (0.05, 0.2):
    for name, fn in methods.items():
        res = [train(fn, lr, seed=s) for s in range(SEEDS)]
        fin = np.array([r[0] for r in res]); rc = [r[1] for r in res if r[1] is not None]
        print(f"   lr={lr:<4} {name:28s} final {fin.mean():.4f} +/- {fin.std(ddof=1):.4f}   reached 0.95 at step "
              f"{np.mean(rc):.0f}" if rc else f"   lr={lr:<4} {name:28s} final {fin.mean():.4f}  never reached 0.95")
S_ex, E_ex = 500, 0.998
print(f"   stopping rule: the sampled objective has SE = sqrt(E(1-E)/S) = {np.sqrt(E_ex * (1 - E_ex) / S_ex):.4f} at E={E_ex}, S={S_ex}; "
      "stop when the moving average changes by less than 2 SE, and confirm with the exact objective on a tiny circuit.")

# ---- 1b. interior target: minimise (E - 0.8)^2 at S=20; the biased _single estimator has a shifted fixed point ------------------
tgt = 0.8
def make_loss_grad(method, S, two_batch):
    b = psc.BranchingSimulator(num_samples=S, diff_method=method)
    Eh = lambda th, k: b.expval_all(b.build_circuit(c3, [th[0:1], th[1:2], th[2:3]]), x_b, k)[1]
    if two_batch:
        return jax.grad(lambda th, k: (Eh(th, jax.random.fold_in(k, 0)) - tgt) * (Eh(th, jax.random.fold_in(k, 1)) - tgt))
    return jax.grad(lambda th, k: (Eh(th, k) - tgt) ** 2)

def train_min(grad_fn, lr=0.05, steps=400, seed=0):
    th = jnp.array([0.3, -0.4, 0.8]); opt = optax.adam(lr); st = opt.init(th); key = jax.random.key(seed)
    @jax.jit
    def step(th, st, k):
        g = grad_fn(th, k); u, st = opt.update(g, st); return optax.apply_updates(th, u), st
    for i in range(steps):
        key, k = jax.random.split(key); th, st = step(th, st, k)
    return float(exact_E(th))

print(f"\n1b. minimise (E_hat - {tgt})^2, S=20, Adam lr 0.05, 400 steps, 6 seeds: final EXACT E (target {tgt}); mean +/- sd")
for name, m, two in (("param_shift_inf, single batch", "param_shift_inf", False),
                     ("param_shift_single, single batch", "param_shift_single", False),
                     ("param_shift_single, two-batch product", "param_shift_single", True),
                     ("param_shift_filter, single batch", "param_shift_filter", False)):
    fin = np.array([train_min(make_loss_grad(m, 20, two), seed=sd) for sd in range(6)])
    print(f"   {name:40s} E = {fin.mean():.4f} +/- {fin.std(ddof=1):.4f} (SE {fin.std(ddof=1) / np.sqrt(len(fin)):.4f})")

# ---- 2. failure cases with the real messages ---------------------------------------------------------------------------
print("\n2. failure cases (real messages):")
comp = sv.build_circuit(psc.DiscretePCircuit([psc.PNOT(i) for i in range(4)]), [jnp.array([0.1])] * 4)
try:
    jax.jit(sv.density)(comp, jnp.zeros(16).at[0].set(1.0))
except Exception as e:
    print("   jax.jit(sv.density):", type(e).__name__, "-", str(e)[:75], "...  (fix: equinox.filter_jit)")
try:
    sv.build_circuit(psc.DiscretePCircuit([psc.PNOT(0), psc.PNOT(1)]), [jnp.array([0.0])])
except Exception as e:
    print("   len(thetas) != len(gates):", type(e).__name__, "-", str(e))
try:
    psc.BranchingSimulator().build_circuit(psc.DiscretePCircuit([psc.PISING([0, 1])]), [jnp.zeros(5)])
except Exception as e:
    print("   PISING + BranchingSimulator:", type(e).__name__, "-", str(e)[:85])
rng = np.random.default_rng(0); d = 288
q = rng.normal(size=d); Kv = rng.normal(size=(8, d)); Kv[3] += 0.8 * q
wts = np.exp(Kv @ q / np.sqrt(d) - (Kv @ q / np.sqrt(d)).max()); wts /= wts.sum()
print(f"   saturated softmax test: max weight {wts.max():.6f}; a TV test against this is vacuous (error 0.0000 at every N)")

# ---- 3. AffineGaussianSimulator.condition with hand arithmetic -------------------------------------------------------
gs = psc.AffineGaussianSimulator()
gates = [psc.AffineGaussianGate(sites=[0], dims=(1,)), psc.AffineGaussianGate(sites=[0, 1], dims=(1, 1))]
th = [{"A": jnp.eye(1), "b": jnp.zeros(1), "log_var": jnp.zeros(1)},
      {"A": jnp.array([[1.0, 0.0], [2.0, 0.0]]), "b": jnp.zeros(2), "log_var": jnp.log(jnp.array([1e-6, 0.25]))}]
comp = gs.build_circuit(psc.HybridPCircuit(gates), th)
post = gs.condition(comp, {1: jnp.array([2.0])}, initial_continuous=jnp.zeros(2))
print("\n3. condition(comp, {site: value}, initial_continuous=...): x0 ~ N(0,1), x1 = 2 x0 + N(0, 0.25), observe x1 = 2")
print("   posterior mean", np.array(post.mean).round(4), " by hand 2*2/4.25 =", round(4 / 4.25, 4))
print("   posterior covariance", np.array(post.covariance).round(4).tolist(), " by hand 1 - 4/4.25 =", round(1 - 4 / 4.25, 4))
print("   observed sites:", np.array(post.observed_sites).tolist(), "(keys of the observation dict)")

# ---- 4. DFG nested as a Site factor ------------------------------------------------------------------------------------
BIT = jax.ShapeDtypeStruct((), jnp.int32); DRIVE = jax.ShapeDtypeStruct((), jnp.float32)

class Coin(AbstractReferenceFactor):
    input_ports: dict = eqx.field(static=True); output_spec: jax.ShapeDtypeStruct = eqx.field(static=True)
    def __init__(self): self.input_ports, self.output_spec = {}, BIT
    def init_params(self, key): return {"bias": jnp.array(0.4)}
    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        o = jax.random.bernoulli(key, sig(params["bias"])).astype(jnp.int32); return (o, None) if return_aux else o

class Cond(AbstractReferenceFactor):
    input_ports: dict = eqx.field(static=True); output_spec: jax.ShapeDtypeStruct = eqx.field(static=True)
    def __init__(self): self.input_ports, self.output_spec = {"drive": DRIVE}, BIT
    def init_params(self, key): return {"w": jnp.array(2.0), "b": jnp.array(-1.0)}
    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        o = jax.random.bernoulli(key, sig(params["w"] * inputs["drive"] + params["b"])).astype(jnp.int32)
        return (o, None) if return_aux else o

to_drive = lambda outs: {"drive": outs[0].astype(jnp.float32)}
inner = DFG((Site("coin", Coin(), (), (), "src", None, None), Site("out", Cond(), ("coin",), to_drive, "cond", None, None)), {}, "out")
outer = DFG((Site("inner", inner, (), (), "inner_params", None, None), Site("final", Cond(), ("inner",), to_drive, "cond2", None, None)), {}, "final")
pr = outer.init_params(jax.random.key(0))
print("\n4. nested DFG: params tree keys", list(pr), "-> inner keys", list(pr["inner_params"]))
draws = jax.vmap(lambda k: outer.sample(k, {}, pr))(jax.random.split(jax.random.key(1), 40000))
pc = float(sig(0.4)); p1 = (1 - pc) * float(sig(-1.0)) + pc * float(sig(1.0))
exact = (1 - p1) * float(sig(-1.0)) + p1 * float(sig(1.0))
print(f"   P(final=1): sampled {float(draws.mean()):.3f}  by hand {exact:.3f} (inner P(out=1) = {p1:.4f})")

# ---- 5. pdit gate tables from a run ---------------------------------------------------------------------------------------
c = sv.build_circuit(psc.DiscretePCircuit([psc.PditShift(0, dims=3)]), [jnp.array([20.0])])
print("\n5. PditShift(dims=3), theta=+20: state i ->", [int(np.argmax(np.array(sv.density(c, jnp.zeros(3).at[i].set(1.0))))) for i in range(3)])
c = sv.build_circuit(psc.DiscretePCircuit([psc.PditSWAP([0, 1], dims=3)]), [jnp.array([20.0])])
tab = {(i, j): divmod(int(np.argmax(np.array(sv.density(c, jnp.zeros(9).at[3 * i + j].set(1.0))))), 3) for i in range(3) for j in range(3)}
print("   PditSWAP(dims=3), theta=+20: (i,j) -> (j,i) for all 9 pairs:", all(v == (j, i) for (i, j), v in tab.items()))

# ---- 6. Torx versus THRML on the same independent-p-bit dot-product estimator ------------------------------------------------
from z1_attention import Z1Config, dot_B
d, N, R = 288, 1000, 20
q = rng.normal(size=d); k = 0.5 * q + rng.normal(size=d)
sq, sk = abs(q).max() * 1.05, abs(k).max() * 1.05
means = np.concatenate([q / sq, k / sk])
thetas = [jnp.array([2 * np.arctanh(m)]) for m in means]
circ = psc.DiscretePCircuit([psc.PNOT(i) for i in range(2 * d)])
bs = psc.BranchingSimulator(num_samples=N); compb = bs.build_circuit(circ, thetas); x0 = jnp.zeros(2 * d, jnp.int32)
tx = []
for r in range(R):
    s = 2.0 * np.asarray(bs.sample(compb, x0, jax.random.key(r)), np.float32) - 1.0
    tx.append(float(np.mean(np.sum(s[:, :d] * s[:, d:], axis=1))) * sq * sk)
th_ = [dot_B(jax.random.key(1000 + r), Z1Config(bias_bits=16), q.astype(np.float32), k.astype(np.float32), N) for r in range(R)]
ex = float(q @ k)
print(f"\n6. same estimator, N={N}, {R} repeats, exact q.k = {ex:.2f}:")
print(f"   Torx  PNOT x576 : mean {np.mean(tx):.2f}  sd {np.std(tx, ddof=1):.2f}")
print(f"   THRML spins x576: mean {np.mean(th_):.2f}  sd {np.std(th_, ddof=1):.2f}   (formula sd about {sq * sk * np.sqrt(np.sum(1 - (q / sq * k / sk) ** 2) / N):.2f})")

import math
a_, b_ = np.array(tx), np.array(th_)
se = math.sqrt(a_.var(ddof=1) / len(a_) + b_.var(ddof=1) / len(b_))
tstat = (a_.mean() - b_.mean()) / se
pval = math.erfc(abs(tstat) / math.sqrt(2))                 # normal approximation to the Welch t-test
f_ratio = a_.var(ddof=1) / b_.var(ddof=1)
print(f"   Welch test on the two sets of means: t = {tstat:.2f}, p = {pval:.2f}; variance ratio Torx/THRML = {f_ratio:.2f} "
      f"(20 repeats each: this fails to reject equality, it does not prove equivalence)")
