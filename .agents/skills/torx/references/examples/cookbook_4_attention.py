"""Cookbook 4: p-bit attention estimator, variance formula, non-degenerate softmax test.

Independent +/-1 p-bits with means m: E[s_a s_b] = m_a m_b, Var(per sample) = sum_i (1 - (m_a_i m_b_i)^2).
So   std(q.k estimate) = s_q s_k sqrt( sum_i (1 - (m_a_i m_b_i)^2) / N ).
Sampling here uses numpy Bernoulli (same law as PNOT p-bits, checked in torx_learn.py).
Run anywhere (numpy only).
"""
import numpy as np

rng = np.random.default_rng(0)
d, K = 288, 8
q = rng.normal(size=d)
qh = q / np.linalg.norm(q)
c = np.linspace(-1.5, 1.5, K) * np.sqrt(d)          # target logits in [-1.5, 1.5]
Kmat = rng.normal(size=(K, d)) * 0.3
# rescale each key's component along q so that q.k_j / sqrt(d) equals the target logit c_j/sqrt(d)... (see print)
for j in range(K):
    along = Kmat[j] @ qh
    Kmat[j] += (c[j] - along * np.linalg.norm(q)) / np.linalg.norm(q) * qh
logits = Kmat @ q / np.sqrt(d)
true_w = np.exp(logits - logits.max()); true_w /= true_w.sum()
print("true logits", logits.round(2))
print("true weights", true_w.round(3), " (max weight %.2f: not saturated)" % true_w.max())

sq, sk = np.abs(q).max() * 1.05, np.abs(Kmat).max() * 1.05
ma, mb = q / sq, Kmat / sk


def scores(N):
    """One q block shared across keys, one independent p-bit block per key."""
    sa = np.where(rng.random((N, d)) < (1 + ma) / 2, 1.0, -1.0)
    out = np.empty(K)
    for j in range(K):
        sb = np.where(rng.random((N, d)) < (1 + mb[j]) / 2, 1.0, -1.0)
        out[j] = (sa * sb).sum(1).mean() * sq * sk
    return out / np.sqrt(d)


def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


# 1. variance formula vs measurement for one key
j, N, R = 3, 1000, 200
pred = sq * sk * np.sqrt(np.sum(1 - (ma * mb[j]) ** 2) / N)
meas = np.std([(np.where(rng.random((N, d)) < (1 + ma) / 2, 1.0, -1.0) *
                np.where(rng.random((N, d)) < (1 + mb[j]) / 2, 1.0, -1.0)).sum(1).mean() * sq * sk
               for _ in range(R)])
print(f"std of q.k estimate at N={N}: formula {pred:.2f}  measured {meas:.2f}  (R={R})")

# 2. TV(softmax(true), softmax(estimated)) vs N, 20 repeats, mean +/- 95% CI
print(f"\n{'N':>6} {'TV mean':>9} {'+/-95%CI (t=2.093)':>19} {'argmax ok':>10}")
for N in (100, 1000, 10000):
    tv, ok = [], []
    for _ in range(20):
        w = softmax(scores(N))
        tv.append(0.5 * np.abs(w - true_w).sum()); ok.append(w.argmax() == true_w.argmax())
    tv = np.array(tv)
    print(f"{N:>6} {tv.mean():>9.4f} {2.093 * tv.std(ddof=1) / np.sqrt(len(tv)):>9.4f} {np.mean(ok):>10.2f}")

# 3. how many samples for a target score error? std_score ~ sq*sk*sqrt(d_eff/N)/sqrt(d)
d_eff = np.mean([np.sum(1 - (ma * mb[j]) ** 2) for j in range(K)])
for tol in (1.0, 0.3, 0.1):
    Nreq = (sq * sk * np.sqrt(d_eff) / np.sqrt(d) / tol) ** 2
    print(f"logit std <= {tol}: N >= {Nreq:,.0f}")

# 4. bias from quantised bias values (assume an n-bit bias DAC; E[s] = tanh(theta))
for bits in (4, 6, 8):
    step = 2 * 3.0 / (2 ** bits - 1)
    th = np.round(np.arctanh(np.clip(ma, -0.999, 0.999)) / step) * step
    dm = np.tanh(th) - ma
    print(f"{bits}-bit DAC (range +/-3): max |dm| {np.abs(dm).max():.4f}, "
          f"bias in first logit {float((np.tanh(th) * sq * mb[0] * sk).sum() / np.sqrt(d) - logits[0]):+.3f}")

# 5. finite-sample TV floor for a histogram (the nb06 0.0634 'TV' is mostly this)
for Ncat, Ns in ((256, 6000), (256, 60000)):
    p = np.ones(Ncat) / Ncat
    tvs = [0.5 * np.abs(rng.multinomial(Ns, p) / Ns - p).sum() for _ in range(50)]
    print(f"TV floor, {Ncat} states, {Ns} samples, exact sampler: {np.mean(tvs):.4f}")

# 6. do key scores that share the q block have correlated errors?  (400 repeats, N=300)
errs = np.array([scores(300) - (Kmat @ q / np.sqrt(d)) for _ in range(400)])
corr = np.corrcoef(errs.T)
off = corr[~np.eye(K, dtype=bool)]
print(f"score-error correlation across keys (shared q block, 400 repeats, N=300): mean off-diagonal {off.mean():+.3f}, "
      f"min {off.min():+.3f}, max {off.max():+.3f}")
