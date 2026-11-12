"""Unit test for the score-matching Hessian sign (errata for Song & Kingma 2021, Eq. 8 / 15 / 17).

For an energy E (p ~ exp(-E)) the implicit score-matching loss is  0.5*|dE/dx|^2 - d2E/dx2.
The arXiv v2 text prints '+ d2E/dx2'. Test on a 1D Gaussian N(0, s2): E = a x^2 / 2, the true a is 1/s2.

Part 1 is algebra on the closed form. Part 2 runs the REAL `loss_ssm` of continuous_ebm_zoo.py (Hessian-vector
product through jax.jvp) on E = a|x|^2/2 with x ~ N(0, s2 I) in 2D, with the energy swapped for the quadratic one.

Run:  cd /tmp && python -I <this file>      (needs numpy, jax; exit code 0 = minus recovers 1/s2, plus does not)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

rng = np.random.default_rng(0)
s2 = 0.5
x = rng.normal(0.0, np.sqrt(s2), 400_000)
a = np.linspace(0.05, 6.0, 4000)

ok = True
for sign, name in [(-1, "minus (correct for an energy)"), (+1, "plus  (as printed in the tutorial)")]:
    loss = 0.5 * a**2 * np.mean(x**2) + sign * a  # 0.5 E_data[(dE/dx)^2] + sign * E[d2E/dx2]
    a_hat = a[np.argmin(loss)]
    print(f"{name}: argmin a = {a_hat:.3f}   true 1/s2 = {1/s2:.3f}")
    if sign == -1:
        ok &= abs(a_hat - 1 / s2) < 0.02
    else:
        ok &= abs(a_hat - 1 / s2) > 0.5
# ---- Part 2: the real SSM loss code -------------------------------------------------------------
import jax
import jax.numpy as jnp
import continuous_ebm_zoo as z

z.energy = lambda ps, xx: 0.5 * ps * jnp.sum(xx**2, -1)  # ps is the scalar precision a
X = jnp.array(rng.normal(0.0, np.sqrt(s2), (20000, 2)), dtype=jnp.float32)
V = jax.random.key(1)
grid = jnp.linspace(0.05, 6.0, 120)
for sg, name in [(-1.0, "SSM_SIGN=-1"), (+1.0, "SSM_SIGN=+1")]:
    z.SSM_SIGN = sg
    losses = jnp.array([z.loss_ssm(g, X, V) for g in grid])
    a_hat = float(grid[int(jnp.argmin(losses))])
    print(f"real loss_ssm, {name}: argmin a = {a_hat:.3f}   true 1/s2 = {1/s2:.3f}")
    ok &= (abs(a_hat - 1 / s2) < 0.1) if sg < 0 else (abs(a_hat - 1 / s2) > 0.5)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
