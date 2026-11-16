# Temporary diversity intervention: exploratory protocol

16 November 2026. GPT-6.1 Sol Ultra reviewed the proposed change before
training. This selected-corpus follow-up investigates whether a short-lived
training-only diversity penalty changes the known 7043 collapse. It is not a
new general-language, physical-hardware, or six-corpus promotion claim.

Corpus 227; permutations 7043, 7019, 7027; model seeds 251,257,263,269;
arms baseline, early-bank, late-bank, early-song, for 48 matched runs. 7043
tests rescue; 7019 guards against damaging a previously successful setup;
7027 shares 7043's initial bank parameters yet had a different outcome.
Song controls whether targeting bank specifically matters. The eight initial
permutations and six-corpus results are already observed: this is posthoc
development, not independent confirmatory evaluation.

Each run retains F2 ranking and 400 lexical updates plus a fixed lexical400
handoff to 100 joint updates with corrected F3. During lexical updates only,
add `0.1*w(u)*KL(q_target || Uniform(4))` once per update. Compute q stably
from the target's two-spin unary scores and use no labels, occurrence weights,
extra random draws or optimizer reset. Early arms use bank or song with
`w(u)=1` for `0<=u<50`, `(100-u)/50` for `50<=u<100`, and zero otherwise;
late-bank uses `w(u-100)`. Baseline is zero. Bonus is identically zero for
the entire joint stage, including its lexical auxiliary loss. It changes
training objective, not physical energy. Shannon entropy here is a
mathematical quantity, not hardware heat or physical-chip entropy.

Outcomes are measured at fixed lexical400 and joint100: original joint sense
>=10/12, AUC>=0.75, mean context gain>=0.03, per-family counts, anchor/context
TV, pure ranking loss and target entropy. Retain lexical50/100/150/200 plus
other original checkpoints to measure the timing. The value 0.1 is a fixed
engineering choice, not tuned to validation. Knowing bank is the problematic
token means the method is not fully label-free.

Practical rescue criterion: early-bank passes >=3/4 seeds on 7043, retains
4/4 on 7019, and is >=2/4 on 7027. Manipulation check: median paired bank
entropy increase at lexical100 >=0.10 nat relative to baseline. Timing:
early-bank exceeds late-bank by >=1 seed on 7043; equal weight area does not
imply equal gradient force. If early-song matches early-bank, targeting
specificity remains unestablished. High entropy without correct F/R and AUC
does not count as semantic rescue. Report full results even when checks fail;
none is a statistical significance test, and no thresholds are retuned.

Preflight: finite-difference KL gradients, zero gradient at uniform, signed
axis symmetry, target-only penalty gradients, schedule endpoints/equal area,
zero-strength baseline parity, no bonus in joint, checkpoint/handoff and
source/data hashes. 48 sequential runs, 45-minute batch deadline and 2 GiB
process working-set snapshot check. No hardware/cloud run, reserved test or
new training data. Failures remain recorded and no replacement is silently
added.
