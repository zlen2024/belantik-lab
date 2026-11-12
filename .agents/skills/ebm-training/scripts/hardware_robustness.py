"""How much parameter error can a trained Boltzmann machine tolerate? (hardware-aware evaluation)

Trains the 8+6 spin RBM of thrml_rbm_negative_phase.py with long-chain MLE, then measures the EXACT KL(p_data || p_model)
after perturbing the parameters the way hardware does (Torx skill, references/hardware-facts.md):
  * quantisation of biases and couplings to n bits over [-R, R], R = max |param|   (X0 uses 8-bit DACs [WP p20]; Z1 DAC unknown)
  * per-site calibration error: about 5% absolute probability [WP p9].  For a spin with P(+1)=sigmoid(2b) at p=0.5,
    dp/db = 0.5, so 5% in probability ~ delta_b = 0.1; we sweep delta_b of 0.02 .. 0.2 as Gaussian noise on the biases
    (couplings left exact, a deliberate simplification: per-edge error is not documented).
Assumptions [ASSUME]: one shared range R for biases and couplings; uniform quantisation in parameter (logit) space,
not in probability space; the 5% calibration figure is read as the std of an additive Gaussian bias error at p = 0.5;
couplings are not perturbed by noise. This is a sensitivity analysis of ONE small model, not a hardware prediction.

Run from OUTSIDE master/:  cd /tmp && python -I <this file>     (about 1-2 min)
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thrml_rbm_negative_phase as rb  # noqa: E402


def quantise(x, bits, R):
    """Mid-tread uniform quantiser: 2^bits - 1 levels symmetric about 0 (zero is representable), step R / (2^(bits-1) - 1)."""
    step = R / (2 ** (bits - 1) - 1)
    return jnp.round(jnp.clip(x, -R, R) / step) * step


def main():
    (b, J), curve, _ = rb.train("long", 600, 0)
    kl0 = float(rb.exact_kl_j(b, J))
    R = float(max(jnp.abs(b).max(), jnp.abs(J).max()))
    print(f"trained 'long' MLE, exact KL {kl0:.4f}; parameter range R = {R:.2f}")
    print("mid-tread quantisation to n bits (biases and couplings):")
    for bits in [3, 4, 5, 6, 8]:
        kl = float(rb.exact_kl_j(quantise(b, bits, R), quantise(J, bits, R)))
        print(f"  {bits} bits: exact KL {kl:.4f}  (delta {kl - kl0:+.4f})")
    step6 = R / (2 ** 5 - 1)
    ref = []
    for k in jax.random.split(jax.random.key(5), 20):  # reference spread: random sub-step perturbations of the same size as 6-bit rounding
        k1, k2 = jax.random.split(k)
        ref.append(float(rb.exact_kl_j(b + step6 * (jax.random.uniform(k1, b.shape) - 0.5), J + step6 * (jax.random.uniform(k2, J.shape) - 0.5))))
    print(f"reference: 20 random uniform(+-step/2) perturbations of 6-bit size: exact KL {np.mean(ref):.4f} +- {np.std(ref):.4f} (min {np.min(ref):.4f}, max {np.max(ref):.4f})")
    print("Gaussian bias noise (delta_b), mean over 20 draws:")
    key = jax.random.key(0)
    for sd in [0.02, 0.05, 0.1, 0.2]:
        kls = []
        for k in jax.random.split(key, 20):
            kls.append(float(rb.exact_kl_j(b + sd * jax.random.normal(k, b.shape), J)))
        print(f"  delta_b {sd:4.2f}: exact KL {np.mean(kls):.4f} +- {np.std(kls):.4f}")
    print("Training-time mitigation (not run here): quantise inside the loop (straight-through) or inject this noise during training, then re-measure.")


if __name__ == "__main__":
    main()
