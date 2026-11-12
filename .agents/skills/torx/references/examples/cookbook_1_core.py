"""Cookbook 1: gradients, sampling API, pdit theta shape. Run from OUTSIDE master/."""
import jax, jax.numpy as jnp, numpy as np
from torx import psc

# ---- 1. gradient of an expectation: jit(grad) over build_circuit + expval_all -------------
circ = psc.DiscretePCircuit([psc.PNOT(0), psc.PCNOT([0, 1]), psc.PNOT(1)])
th0 = [jnp.array([0.3]), jnp.array([-0.4]), jnp.array([0.8])]
sv = psc.StateVectorSimulator()
x_sv = jnp.zeros(4).at[0].set(1.0)
x_b = jnp.zeros(2, dtype=jnp.int32)

def exact_loss(th):
    return sv.expval_all(sv.build_circuit(circ, th), x_sv)[1]   # E[bit 1]

g_exact = jax.grad(exact_loss)(th0)
print("exact grad     ", [float(g[0]) for g in g_exact])

for m in ("param_shift_inf", "param_shift_single", "param_shift_filter"):
    bsim = psc.BranchingSimulator(num_samples=4000, diff_method=m)

    def samp_loss(th, key):
        return bsim.expval_all(bsim.build_circuit(circ, th), x_b, key)[1]   # build INSIDE jit is fine

    gj = jax.jit(jax.grad(samp_loss))
    gs = np.array([[float(g[0]) for g in gj(th0, jax.random.key(i))] for i in range(40)])
    print(f"{m:20s} mean {gs.mean(0).round(3)}  sd {gs.std(0).round(3)}")

# ---- 2. only expval_all is differentiable; sample() is not --------------------------------
bsim = psc.BranchingSimulator(num_samples=8)
comp = bsim.build_circuit(circ, th0)
print("sample shape", bsim.sample(comp, x_b, jax.random.key(0)).shape)

# ---- 3. circuit.sample / sample_multiple on the DFG interface -----------------------------
params = th0
out = circ.sample(jax.random.key(1), {"in": x_b}, params)
print("circuit.sample ->", out)
outs = circ.sample_multiple(jax.random.key(2), {"in": x_b}, params, n_samples=5)
print("sample_multiple shape", outs.shape)

# ---- 4. pdit gates: theta shape is (n_branches-1,) and branches follow softmax([0, theta]) -
cyc = psc.PditCycle(0, dims=3)
pc = psc.DiscretePCircuit([cyc])
p = pc.init_params(jax.random.key(0))
print("PditCycle init theta shape", p[0].shape, "num_branches", cyc.num_branches)
theta = [jnp.array([1.0, -0.5])]
dens = sv.density(sv.build_circuit(pc, theta), jnp.array([1.0, 0, 0]))
want = jax.nn.softmax(jnp.array([0.0, 1.0, -0.5]))   # stay, forward, backward
print("density      ", np.array(dens).round(4), " (state 0 start; forward->1, backward->2)")
print("softmax rule ", np.array(want).round(4))

# ---- 5. state-vector index order: site 0 is the most significant bit ----------------------
c2 = psc.DiscretePCircuit([psc.PNOT(0), psc.PNOT(1)])
d2 = sv.density(sv.build_circuit(c2, [jnp.array([20.0]), jnp.array([-20.0])]),
                jnp.array([1.0, 0, 0, 0]))
print("flip site 0 from |00) ->", np.array(d2).round(3), "(index 2 = |10)  => site 0 is MSB)")

# ---- 6. hand arithmetic for the worked example in explain-code.md (Example A) ------------------------------------------------
p_ = 1 / (1 + np.exp(-np.array([0.3, -0.4, 0.8])))
s_ = p_ * (1 - p_)                                          # sigmoid'(theta)
pb1 = p_[0] * p_[1]                                         # P(bit 1 = 1) after the CNOT
Efin = pb1 * (1 - p_[2]) + (1 - pb1) * p_[2]
print(f"hand: p = {p_.round(4)}, sigmoid' = {s_.round(4)}, P(b1) after CNOT {pb1:.4f}, final E {Efin:.4f}, 1-P {1 - pb1:.4f}, 1-2P {1 - 2 * pb1:.4f}, 1-2p3 {1 - 2 * p_[2]:.4f}")
print(f"hand grads: {s_[2] * (1 - 2 * pb1):.4f} {p_[0] * s_[1] * (1 - 2 * p_[2]):.4f} {s_[0] * p_[1] * (1 - 2 * p_[2]):.4f} (theta3, theta2, theta1)")
