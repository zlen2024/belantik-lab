"""Cookbook 7: API coverage. Run from OUTSIDE master/.

Gaussian-circuit gradients, log_var=-inf, draw_circuit, circuit +, to_thetas/to_pcircuit, pdit expval,
sample_with_references / return_aux, DFGInfo + info_key, log_probability, and a custom continuous gate.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")   # draw_circuit prints box-drawing characters; default Windows console codec fails
from typing import Any, Mapping

import equinox as eqx
import jax, jax.numpy as jnp, numpy as np
from torx import AbstractReferenceFactor, DFG, DFGInfo, Site, psc
from torx.psc import AbstractContinuousGate, HybridSites

# ---- 1. gradient through the exact Gaussian simulator ---------------------------------------
gates = [psc.AffineGaussianGate(sites=[0], dims=(1,)),
         psc.AffineGaussianGate(sites=[0, 1], dims=(1, 1))]
circ = psc.HybridPCircuit(gates)
gs = psc.AffineGaussianSimulator()


def post_mean(A21, log_var_noise):
    th = [{"A": jnp.eye(1), "b": jnp.zeros(1), "log_var": jnp.zeros(1)},
          {"A": jnp.array([[1.0, 0.0], [A21, 0.0]]), "b": jnp.zeros(2),
           "log_var": jnp.stack([jnp.log(1e-6), log_var_noise])}]
    comp = gs.build_circuit(circ, th)
    return gs.condition(comp, {1: jnp.array([2.0])}, initial_continuous=jnp.zeros(2)).mean[0]

g = jax.grad(post_mean, argnums=(0, 1))(2.0, jnp.log(0.25))
eps = 1e-3
fd = [(post_mean(2.0 + eps, jnp.log(0.25)) - post_mean(2.0 - eps, jnp.log(0.25))) / (2 * eps),
      (post_mean(2.0, jnp.log(0.25) + eps) - post_mean(2.0, jnp.log(0.25) - eps)) / (2 * eps)]
print("d posterior-mean / d(A21, log_var): autodiff", [round(float(a), 4) for a in g],
      " finite diff", [round(float(a), 4) for a in fd])

# log_var = -inf (deterministic channel): what happens to the gradient?
def with_inf(A21, lv):
    th = [{"A": jnp.eye(1), "b": jnp.zeros(1), "log_var": jnp.zeros(1)},
          {"A": jnp.array([[1.0, 0.0], [A21, 0.0]]), "b": jnp.zeros(2), "log_var": jnp.array([lv, lv])}]
    return gs.propagate(gs.build_circuit(circ, th), jnp.zeros(2)).mean[1] + \
        gs.propagate(gs.build_circuit(circ, th), jnp.zeros(2)).covariance[1, 1]
for lv in (-jnp.inf, -30.0):
    print(f"log_var={float(lv):>6}: grad wrt A21 =", float(jax.grad(with_inf)(2.0, lv)),
          "  grad wrt log_var =", float(jax.grad(with_inf, argnums=1)(2.0, lv)))

# ---- 2. expval on the Gaussian simulator and on a pdit -----------------------------------------
th_ok = [{"A": jnp.eye(1), "b": jnp.zeros(1), "log_var": jnp.zeros(1)},
         {"A": jnp.array([[1.0, 0.0], [2.0, 0.0]]), "b": jnp.zeros(2),
          "log_var": jnp.log(jnp.array([1e-6, 0.25]))}]
print("AffineGaussianSimulator.expval(site 1) =",
      float(gs.expval(gs.build_circuit(circ, th_ok), jnp.zeros(2), 1)[0]), "(signature: expval(circuit, initial_continuous, site); mean of x1 = 0)")
cyc = psc.DiscretePCircuit([psc.PditCycle(0, dims=3)])
sv = psc.StateVectorSimulator()
ev = sv.expval(sv.build_circuit(cyc, [jnp.array([20.0, -20.0])]), jnp.array([1.0, 0, 0]), 0)
print("pdit expval after a forward shift from state 0:", float(jnp.ravel(ev)[0]), "(= mean INDEX 1.0, not a probability)")

# ---- 3. drawing, circuit addition, theta recovery ----------------------------------------------
c_a = psc.DiscretePCircuit([psc.PNOT(0), psc.PCNOT([0, 1])])
c_b = psc.DiscretePCircuit([psc.PNOT(1)])
c_ab = c_a + c_b
print("\n" + str(psc.draw_circuit(c_ab, output=False)))
th = [jnp.array([0.3]), jnp.array([-0.4]), jnp.array([0.8])]
bsim = psc.BranchingSimulator(num_samples=4)
comp = bsim.build_circuit(c_ab, th)
print("to_thetas round trip:", [round(float(t[0]), 3) for t in comp.to_thetas()],
      " to_pcircuit gates:", len(comp.to_pcircuit(c_ab).gates))

# ---- 4. factors: aux, references, log_probability, DFGInfo/info_key -----------------------------
BIT = jax.ShapeDtypeStruct((), jnp.int32)

class TempBit(AbstractReferenceFactor):
    """Bernoulli(sigmoid(w * temperature_scale)), scale read from runtime `info`."""
    input_ports: dict = eqx.field(static=True)
    output_spec: jax.ShapeDtypeStruct = eqx.field(static=True)

    def __init__(self):
        self.input_ports, self.output_spec = {}, BIT

    def init_params(self, key):
        return {"w": jnp.array(1.0)}

    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        scale = 1.0 if info is None else info["scale"]
        p = jax.nn.sigmoid(params["w"] * scale)
        out = jax.random.bernoulli(key, p).astype(jnp.int32)
        return (out, {"p": p}) if return_aux else out

f = TempBit()
pr = f.init_params(jax.random.key(0))
out, aux = f.sample(jax.random.key(1), {}, pr, return_aux=True)
print("\nreturn_aux ->", int(out), {k: round(float(v), 3) for k, v in aux.items()})
main, auxes = f.sample_with_references(jax.random.key(2), {}, pr, n_references=3)
print("sample_with_references: aux leaf leading axis =", auxes["p"].shape, "(= n_references + 1)")

graph = DFG((Site("b", f, (), (), "k", "knobs", None),), {}, "b")
params = graph.init_params(jax.random.key(0))
for scale in (0.0, 3.0):
    info = DFGInfo(expose_site_outputs=False, entries={"knobs": {"scale": scale}})
    draws = jax.vmap(lambda k: graph.sample(k, {}, params, info))(jax.random.split(jax.random.key(3), 20000))
    print(f"info scale={scale}: P(1) sampled {float(draws.mean()):.3f}  exact {float(jax.nn.sigmoid(scale)):.3f}")

# ---- 5. a custom continuous gate (AR(1)/OU step) with an analytic test -------------------------------
def _sites(x):
    return x if isinstance(x, dict) else {"discrete": [], "continuous": [x] if not isinstance(x, list) else x}


class OUStep(AbstractContinuousGate[dict[str, jax.Array], tuple[int, ...]]):
    """x' = x + a (mu - x) + sigma * N(0,1).  Not affine-simulator compatible unless you add affine_parameters."""
    _draw_label = "OU"
    sites: HybridSites = eqx.field(converter=_sites)
    dims: tuple[int, ...]

    def init_params(self, key):
        return {"a": jnp.array(0.1), "mu": jnp.array(1.0), "sigma": jnp.array(0.3)}

    def sample(self, key, inputs: Mapping[str, Any], params, info=None, site_info=None, return_aux=False):
        x = inputs["continuous"]
        out = x + params["a"] * (params["mu"] - x) + params["sigma"] * jax.random.normal(key, x.shape)
        return (out, None) if return_aux else out

n = 20
oc = psc.HybridPCircuit([OUStep(sites=[0], dims=(1,))], reps=n)
pp = oc.init_params(jax.random.key(0))
st = {"discrete": jnp.zeros((0,), jnp.int32), "continuous": jnp.zeros(1)}
outs = oc.sample_multiple(jax.random.key(5), st, pp, n_samples=40000)["continuous"][:, 0]
a, mu, s = 0.1, 1.0, 0.3
mean_exact = mu * (1 - (1 - a) ** n)
var_exact = s ** 2 * (1 - (1 - a) ** (2 * n)) / (1 - (1 - a) ** 2)
print(f"\ncustom OU gate, reps={n}: mean {float(outs.mean()):.3f} (exact {mean_exact:.3f}), "
      f"var {float(outs.var()):.4f} (exact {var_exact:.4f})")

# ---- 6. exact log_probability from AbstractMatrixFactor (finite-state factor, nb15) ----------------
from torx import AbstractMatrixFactor


class TinyCategorical(AbstractMatrixFactor):
    """3-state pdit, no inputs, exact log-probabilities from a (1, 3) log-probability matrix."""
    input_states: dict
    output_states: dict

    def __init__(self):
        self.input_states, self.output_states = {}, {"x": jnp.arange(3)}

    def init_params(self, key):
        return {"logits": jnp.array([0.2, 1.0, -0.5])}

    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        row = self.get_log_probability_matrix(params, info, site_info)[0]
        out = self.get_nth_output_state(jax.random.categorical(key, row))
        return (out, None) if return_aux else out

    def get_log_probability_matrix(self, params, info=None, site_info=None):
        logits = params["logits"]
        return (logits - jax.scipy.special.logsumexp(logits))[None, :]

cat = TinyCategorical()
cp = cat.init_params(jax.random.key(0))
lp = [float(cat.log_probability({}, {"x": jnp.array(i)}, cp)) for i in range(3)]
print("\nlog_probability:", np.round(lp, 4), " softmax:", np.round(np.log(jax.nn.softmax(cp["logits"])), 4))

# ---- 7. pathwise gradients through HybridPCircuit.sample, log_var=-inf on the SAMPLING route ----------------
st0 = {"discrete": jnp.zeros((0,), jnp.int32), "continuous": jnp.array([0.5])}
ag = psc.HybridPCircuit([psc.AffineGaussianGate(sites=[0], dims=(1,))])
for lv in (-jnp.inf, -30.0):
    thg = {"A": jnp.eye(1), "b": jnp.zeros(1), "log_var": jnp.array([lv])}
    gr = jax.grad(lambda t: ag.sample(jax.random.key(0), st0, [t])["continuous"][0])(thg)
    print(f"\nHybridPCircuit.sample, log_var={float(lv)}: dA={float(gr['A'][0, 0]):.3f} db={float(gr['b'][0]):.3f} "
          f"dlog_var={float(gr['log_var'][0])}   (NaN route when -inf)" if lv == -jnp.inf else
          f"HybridPCircuit.sample, log_var={float(lv)}: dA={float(gr['A'][0, 0]):.3f} db={float(gr['b'][0]):.3f} "
          f"dlog_var={float(gr['log_var'][0]):.1e}")
hc = psc.HybridPCircuit([psc.PditCycle(0, dims=3), psc.MixtureGaussianGate(sites=(0, 0), dims=(1,), num_components=3)])
hp = [jnp.array([0.5, -0.5]), {"means": jnp.array([[2.0], [0.0], [-2.0]]), "log_vars": jnp.full((3, 1), -2.0)}]
hg = jax.grad(lambda p: hc.sample(jax.random.key(0), {"discrete": jnp.array([0], jnp.int32), "continuous": jnp.zeros(1)}, p)["continuous"][0])(hp)
print("pathwise grads through a hybrid sample: d/d PditCycle theta =", np.array(hg[0]), "(the discrete branch choice is NOT differentiable),",
      " d/d means =", np.array(hg[1]["means"]).ravel(), "(only the selected component)")

# condition with a zero-variance observed site needs jitter on some circuits; with jitter it is finite:
gs2 = psc.AffineGaussianSimulator()
det = [psc.AffineGaussianGate(sites=[0], dims=(1,)), psc.AffineGaussianGate(sites=[0, 1], dims=(1, 1))]
thd = [{"A": jnp.eye(1), "b": jnp.zeros(1), "log_var": jnp.zeros(1)},
       {"A": jnp.array([[1.0, 0.0], [2.0, 0.0]]), "b": jnp.zeros(2), "log_var": jnp.array([-jnp.inf, -jnp.inf])}]
cd = gs2.condition(gs2.build_circuit(psc.HybridPCircuit(det), thd), {1: jnp.array([2.0])},
                   initial_continuous=jnp.zeros(2), jitter=1e-6)
print("condition(..., jitter=1e-6) on a deterministic observed site:", np.array(cd.mean))

# JumpDiffusionGate: x' = x + N(0, exp(diff_log_var)) + j (jump_mean + N(0, exp(jump_log_var))), j = discrete control in {0,1}
jc = psc.HybridPCircuit([psc.PNOT(0), psc.JumpDiffusionGate(sites=(0, 0), dims=(1,))])
jp = [jnp.array([0.0]), {"diff_log_var": jnp.array([np.log(0.01)]), "jump_mean": jnp.array([3.0]), "jump_log_var": jnp.array([np.log(0.04)])}]
outs = jc.sample_multiple(jax.random.key(0), {"discrete": jnp.array([0], jnp.int32), "continuous": jnp.zeros(1)}, jp, n_samples=20000)
j = np.array(outs["discrete"][:, 0]); x = np.array(outs["continuous"][:, 0])
print(f"JumpDiffusionGate (PNOT theta=0 fires j with p=0.5): P(j=1)={j.mean():.3f}, mean x | j=0 {x[j == 0].mean():+.3f}, "
      f"mean x | j=1 {x[j == 1].mean():+.3f} (jump_mean 3.0), var x | j=0 {x[j == 0].var():.4f} (0.01)")


# ---- 8. tractable factors: exact log p for score-function code, and a second jit failure mode ---------------------------------------------
from torx import DeterministicFactor, AbstractFiniteStateSpaceFactor, AbstractHasLogProbability
xor = DeterministicFactor(fn=lambda inp, si: jnp.bitwise_xor(inp["a"], inp["b"]),
                          input_ports={"a": BIT, "b": BIT}, output_spec=BIT)
o_ = xor.sample(jax.random.key(0), {"a": jnp.array(1), "b": jnp.array(0)}, None)
lp_ok = float(xor.log_probability({"a": jnp.array(1), "b": jnp.array(0)}, jnp.array(1), None))
lp_bad = float(xor.log_probability({"a": jnp.array(1), "b": jnp.array(0)}, jnp.array(0), None))
print(f"DeterministicFactor xor: sample {int(o_)}, log p(correct) {lp_ok}, log p(wrong) {lp_bad}")
print("TinyCategorical is AbstractFiniteStateSpaceFactor:", isinstance(cat, AbstractFiniteStateSpaceFactor),
      " AbstractHasLogProbability:", isinstance(cat, AbstractHasLogProbability))
cb_ = psc.DiscretePCircuit([psc.PNOT(0), psc.PNOT(1)])
bb_ = psc.BranchingSimulator(num_samples=10)
comp_ = bb_.build_circuit(cb_, [jnp.array([0.1]), jnp.array([0.2])])
try:
    jax.jit(bb_.expval_all)(comp_, jnp.zeros(2, jnp.int32), jax.random.key(0))
except Exception as e:
    print("jax.jit(bsim.expval_all)(compiled, ...) ->", type(e).__name__, "(fix: equinox.filter_jit, or build inside the jitted function)")
print("eqx.filter_jit(bsim.expval_all) works:", eqx.filter_jit(bb_.expval_all)(comp_, jnp.zeros(2, jnp.int32), jax.random.key(0)).shape)
# score of a tractable factor: d log p(x = 1) / d logits = onehot(1) - softmax(logits)
score_ = np.array(jax.grad(lambda pr: cat.log_probability({}, {"x": jnp.array(1)}, pr))(cp)["logits"])
print("d log p(x=1)/d logits:", np.round(score_, 4), " onehot - softmax:", np.round(np.eye(3)[1] - np.array(jax.nn.softmax(cp["logits"])), 4))
