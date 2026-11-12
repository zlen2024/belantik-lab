"""Regression runner: execute every example from a temporary cwd and assert the numbers quoted in the skill.

Run:  python check_examples.py [--quick] [script_name ...]      (--quick: five fast scripts, no doc-number check)       (from anywhere; it changes cwd itself)
Exit code 0 = all pass. Takes several minutes (JAX compilation, THRML sampling).

Each check is (description, regex, expected, tolerance):
  * the regex may have one capture group and match several times; `expected` is then a list compared element-wise
    (use a scalar for a single first match);
  * a string `expected` asserts that the text matches (for example "nan").
Seeds are fixed, so values reproduce exactly on one machine; tolerances are about 3 standard errors of the stated
Monte Carlo error so that a changed claim fails but a library-version wobble does not.
"""
import os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
F = r"(-?[\d.]+(?:e-?\d+)?)"

CHECKS = {
    "cookbook_1_core.py": [
        ("exact grad 1", r"exact grad\s+\[(-?[\d.]+),", -0.0373, 0.0005),
        ("exact grad 2", r"exact grad\s+\[-?[\d.]+, (-?[\d.]+),", -0.0524, 0.0005),
        ("exact grad 3", r"exact grad\s+\[-?[\d.]+, -?[\d.]+, (-?[\d.]+)\]", 0.1153, 0.0005),
        ("PditCycle softmax stay", r"density\s+\[([\d.]+)", 0.2312, 0.0005),
        ("MSB index", r"-> \[0\. 0\. (1)\.", 1.0, 0.0),
        ("inf mean theta3", r"param_shift_inf\s+mean \[[-\d. ]+ [-\d. ]+ ([\d.]+)\]", 0.115, 0.004),
    ],
    "cookbook_2_hybrid.py": [
        ("posterior mean", r"posterior mean given x1 = 2: \[([\d.]+)\]", 0.941, 0.002),
        ("P(state=1) within 3 SE of 0.8668", r"P\(state=1\) sampled ([\d.]+)", 0.8668, 0.0025),
    ],
    "cookbook_3_factors.py": [("DFG marginal", r"exact ([\d.]+)", 0.546, 0.002)],
    "cookbook_4_attention.py": [
        ("variance formula", r"formula ([\d.]+)", 1.95, 0.03),
        ("measured sd", r"measured ([\d.]+)", 1.81, 0.2),
        ("TV N=100,1000,10000", r"\n\s+(?:100|1000|10000)\s+([\d.]+)\s+[\d.]+\s+[\d.]+", [0.1173, 0.0356, 0.0128], 0.009),
        ("N required", r"N >= ([\d,]+)", None, None),
        ("score-error correlation mean off-diagonal", r"mean off-diagonal ([+-]?[\d.]+)", -0.011, 0.03),
        ("TV floor uniform", r"6000 samples, exact sampler: ([\d.]+)", 0.0822, 0.006),
    ],
    "cookbook_5_gibbs.py": [
        ("one-site kernel both starts", r"start [01]: P\(1\) = ([\d.]+)", [0.8022, 0.8022], 0.0005),
        ("Gibbs TV", r"TV\(sampler, exact\) = ([\d.]+)", 0.0457, 0.008),
        ("floor", r"draws = ([\d.]+)", 0.0468, 0.006),
    ],
    "cookbook_6_gradients.py": [
        ("K=2 table: 15 rows, max|mean-exact| < 0.008",
         r"theta=[+-]\d param_shift_\w+\s+([\d.]+)\s+[\d.]+\s+[\d.]+", "max<0.008", None),
        ("sd ratios ~2", r"ratio=([\d.]+)", [1.92, 1.88, 2.05], 0.25),
        ("filter bias N=20,100,1000", r"max\|bias\| = ([\d.]+)", [0.0057, 0.0002, 0.0001], 0.006),
        ("silent zeros", r"jax.grad through sample\(\): \[([\d.]+),", 0.0, 0.0),
        ("REINFORCE M=1e5 grad3", r"M=100000: mean \[[-\d. ]+ [-\d. ]+ ([\d.]+)\]", 0.1153, 0.0025),
    ],
    "cookbook_7_api.py": [
        ("autodiff vs fd A21", r"autodiff \[(-?[\d.]+),", -0.4152, 0.001),
        ("log_var inf finite via simulator", r"log_var=\s+-inf: grad wrt A21 = ([\d.]+)", 4.0, 0.001),
        ("NaN on sample route", r"log_var=-inf: dA=[\d.]+ db=[\d.]+ dlog_var=(nan)", "nan", None),
        ("clamped finite", r"log_var=-30.0: dA=[\d.]+ db=[\d.]+ dlog_var=([\d.e-]+)", 1.5e-07, 1e-06),
        ("pdit expval is mean index", r"forward shift from state 0: ([\d.]+)", 1.0, 0.0),
        ("OU mean,var", r"mean ([\d.]+) \(exact [\d.]+\), var ([\d.]+) \(exact", None, None),
        ("JumpDiffusion mean|j=1", r"mean x \| j=1 \+?([\d.]+)", 3.0, 0.05),
        ("log_probability", r"log_probability: \[(-?[\d.]+)", -1.3143, 0.0005),
        ("DeterministicFactor log p wrong output", r"log p\(wrong\) (-inf)", "-inf", None),
        ("bsim jit failure mode", r"\) -> (TracerBoolConversionError)", "tracerboolconversionerror", None),
    ],
    "cookbook_8_attention_thrml.py": [
        ("DAC TV 4,6,8,16", r"bias DAC\s+\d+ bits: TV = ([\d.]+)", [0.0592, 0.0207, 0.0094, 0.0123], 0.008),
        ("one-hot mass lam=1,2,4,6", r"\n\s+[1246]\s+([\d.]+)\s+[\d.e+-]+", [0.331, 0.600, 0.918, 0.988], 0.005),
        ("Gibbs one-hot fraction", r"one-hot fraction ([\d.]+)", [0.600, 0.915, 0.989], 0.02),
        ("consecutive identical", r"consecutive samples identical ([\d.]+)", [0.21, 0.76, 0.97], 0.05),
        ("N_eff/N", r"N_eff/N = ([\d.]+)", [0.91, 0.46, 0.15, 0.05], 0.01),
    ],
    "cookbook_9_variance.py": [
        ("single PNOT: Eq.40 pred 0.0148 vs measured", r"param_shift_single\s+mean [\d.]+ sd ([\d.]+)", 0.0150, 0.0015),
        ("inf per-gate pred vs meas", r"_inf\s+pred ([\d.]+) meas ([\d.]+)", None, None),
        ("equal budget sd inf,single,filter,REINFORCE", r"sd ([\d.]+)\s+max\|bias\|", [0.0027, 0.0082, 0.0013, 0.0014], 0.0007),
        ("baseline ratio (M-1)/M", r"uncorrected/exact = ([\d.]+)", 0.950, 0.015),
        ("filter chain sd depth 3,6,10 within 15 percent of 0.0079", r"depth\s+\d+: measured mean sd ([\d.]+)", [0.0069, 0.0080, 0.0077], 0.0012),
        ("K=3 filter sd predicted vs measured (within 10 percent)", r"theta=\[[-\d., ]+\]: predicted \[([\d. ,]+)\]  measured \[([\d. ,]+)\]", "k3pairs", None),
        ("single-gate _filter sd: predicted vs measured", r"predicted ([\d.]+)  measured ([\d.]+)", "filtpairs", None),
        ("filter bias N=20 gate3", r"N=\s+20: mean \[[-\d. ]+ [-\d. ]+ ([\d.]+)\]", 0.0053, 0.001),
        ("K=3 filter bias N=20 (both comps)", r"K=3 gate[^\n]*\n\s+N=\s+20: mean \[[-\d. ]+\]  bias \[(-?[\d.]+)", -0.0125, 0.002),
        ("equal-budget RMSE theta=-4,-2,0,2 (_inf)", r"\n\s+[-+]\d\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+best", [0.00005, 0.00074, 0.00272, 0.00089], 0.0005),
        ("equal-budget best method", r"best: (_\w+)", "winners", None),
        ("depth 6,10 _inf pred vs meas", r"depth\s+\d+: _inf pred ([\d.]+) meas ([\d.]+)", "pairs0.0006", None),
    ],
    "cookbook_10_key_sampling.py": [
        ("TV steps 1,5,20 (12 seeds)", r"\n\s+(?:1|5|20)\s+[\d.]+\s+([\d.]+) \+/-", [0.0139, 0.0079, 0.0062], 0.003),
        ("rho steps 1,5,20", r"\+/- [\d.]+\s+[\d.]+\s+(-?[\d.]+)\s+\d+", [0.527, 0.040, 0.000], 0.03),
        ("leakage ~0.197", r"\n\s+(?:1|5|20)\s+([\d.]+)\s+[\d.]+ \+/-", [0.197, 0.197, 0.197], 0.01),
        ("DAC TV 4,6,8,ideal (12 seeds)", r"leakage [\d.]+  TV ([\d.]+) \+/-", [0.2451, 0.0585, 0.0170, 0.0067], 0.005),
    ],
    "cookbook_11_train.py": [
        ("law prediction 2E(1-E)/S at S=20,100,1000", r"S=\d+: ([\d.]+)", [0.02395, 0.00479, 0.00048], 0.00002),
        ("law vs measured bias, 3-gate S=40 and 160 (all gates within 40 percent)", r"3-gate chain, t=0.8: E=[\d.]+, S=\d+: predicted ([\d.]+), measured bias per gate \[([\d. ]+)\]", "lawpairs", None),
        ("final exact objective x4", r"E = ([\d.]+)\s+theta", [0.9986] * 4, 0.0005),
        ("_single Jensen bias S=20,100,1000 (gate 1)", r"S=\s*\d+ param_shift_single\s+single batch bias \[\s*([\d.]+)", [0.0226, 0.0052, 0.0004], 0.0025),
        ("_inf bias small S=20,100,1000", r"S=\s*\d+ param_shift_inf\s+single batch bias \[\s*(-?[\d.]+)", [0.0, 0.0002, 0.0], 0.0008),
        ("_filter bias gate 3 at S=20", r"S=  20 param_shift_filter\s+single batch bias \[[-\d. ]+ [-\d. ]+ (-?[\d.]+)\s*\]", -0.0040, 0.002),
        ("REINFORCE sd none, bm, loo", r"\n\s+(?:none|batch-mean x M/\(M-1\)|loo)\s+sd ([\d.]+)", [0.0104, 0.0073, 0.0073], 0.002),
        ("score term vs analytic", r"score-function term \[\s*(-?[\d.]+)", -0.1223, 0.004),
        ("pathwise zero", r"pathwise through sample\(\) \[\s*([\d.]+)", 0.0, 0.0),
        ("PDEMUX table", r"PDEMUX\(\[0,1\]\)\s+: (.*)", "00->00 01->00 10->01 11->01", None),
        ("Mix rotation", r"Mix\(theta=pi/2\) rotates \(1,0\) -> \[\s*-?([\d.]+)", 0.0, 0.01),
        ("jit TypeError", r"jax.jit\(sv.density\) -> (TypeError)", "typeerror", None),
        ("jitted n=20 under 200 ms", r"n=20: 2\^n =\s+\d+ states, jitted\s+([\d.]+) ms", 21.0, 20.0),
    ],
    "cookbook_12_study.py": [
        ("interior-target final E: inf, single, single two-batch, filter", r"E = ([\d.]+) \+/-", [0.7985, 0.7367, 0.8035, 0.8167], 0.03),
        ("single batch fixed point shifted below target", r"param_shift_single, single batch\s+E = ([\d.]+)", 0.7367, 0.04),
        ("Welch p-value above 0.05", r"p = ([\d.]+);", 0.96, 0.05),
        ("final objective lr=0.05 (4 methods)", r"lr=0\.05\s+[^\n]*final ([\d.]+)", [0.9885, 0.9882, 0.9884, 0.9885], 0.0015),
        ("final objective lr=0.2 (4 methods)", r"lr=0\.2\s+[^\n]*final ([\d.]+)", [0.9989] * 4, 0.0005),
        ("steps to 0.95 lr=0.2", r"lr=0\.2\s+[^\n]*reached 0\.95 at step (\d+)", [13] * 4, 2),
        ("len(thetas) message", r"ValueError - (expected 2 thetas aligned with circuit.gates, got 1\.)", "expected 2 thetas aligned with circuit.gates, got 1.", None),
        ("saturated max weight", r"max weight ([\d.]+);", 0.999914, 0.00002),
        ("posterior mean/var", r"posterior mean \[([\d.]+)\]", 0.9412, 0.0003),
        ("posterior variance", r"posterior covariance \[\[([\d.]+)", 0.0588, 0.0003),
        ("nested DFG hand vs sampled", r"P\(final=1\): sampled ([\d.]+)", 0.521, 0.012),
        ("PditSWAP all pairs", r"for all 9 pairs: (True)", "true", None),
        ("PditShift", r"state i -> \[(\d, \d, \d)\]", "1, 2, 0", None),
        ("Torx vs THRML mean", r"mean ([\d.]+)\s+sd", [142.24, 142.34], 4.0),
    ],
    "cookbook_13_bias_theory.py": [
        ("per-gate law (all 10 rows within 3 SE + 0.0003 per gate)",
         r"S=\s*(\d+) E=[\d.]+ Var=[\d.]+\s+predicted \[([\d. ]+)\]\s+measured \[([\d. ]+)\]\s+SE up to ([\d.]+)", "law", None),
        ("cubic and exp rows have gate-dependent predictions", r"(?:cubic|exp)\s+S=\s*\d+ E=[\d.]+ Var=[\d.]+\s+predicted \[([\d. ]+)\]", "gatedep", None),
        ("fixed points inf, single, filter at S=20 and 50 (within 0.0025)", r"\n\s+(?:20|50) (?:inf|single|filter)\s+([\d.]+)\s+([\d.]+)", "fixed", None),
        ("_inf last gate predicted vs measured (cubic S=20,40; exp S=20,40) within 0.0012",
         r"(?:cubic|exp)\s+S=\s*\d+ _inf predicted \[[-\d. ]+ (-?[\d.]+)\] measured \[[-\d. ]+ (-?[\d.]+)\]", "infpairs", None),
        ("_filter last gate bias cubic and exp, S=20 and 40", r"(?:cubic|exp)\s+S=\s*\d+ _filter measured \[[-\d. ]+ (-?[\d.]+)\]", [-0.0063, -0.0030, -0.0190, -0.0091], 0.0025),
        ("k-batch remedy: three batches unbiased, single batch biased", r"(?:single batch \(E1\^3\)|three independent batches \(E1 E2 E3\))\s+bias ([-\d. ]+) ", "kbatch", None),
        ("small-S ratio measured/predicted, cubic and exp at S=20", r"ratio measured/predicted ([\d. ]+)", "ratio", None),
        ("chain theta drifts to -infinity", r"theta = \[(-?[\d.]+) ", [-18.0248, -197.0857], 20.0),
        ("Eq. 44 equal runs: parameter shift sd", r"parameter shift \(_inf, S=\d+, x2\)\s+mean -?[\d.]+  sd ([\d.]+)", 0.0104, 0.003),
        ("Eq. 44 equal runs: exact score + baseline beats parameter shift", r"baseline f - mean f\s+mean -?[\d.]+  sd ([\d.]+)", 0.0055, 0.002),
        ("_filter delta-method bias per gate (7 rows, within 3.5 SE + 0.0001)",
         r"_filter predicted \[([-\d. ]+)\]  measured \[([-\d. ]+)\]  SE up to ([\d.]+)\s+\(_single predicted", "filtlaw", None),
        ("_filter variance formula depth-3 chain", r"sd predicted \[[\d. ]+\] \(mean ([\d.]+)\)  measured \[[\d. ]+\] \(mean ([\d.]+)\)", "filtvar", None),
        ("d/da-only parameter shift sd vs formula", r"d/da only \(2 x \d+\)\s+mean -?[\d.]+  sd ([\d.]+)   \(formula ([\d.]+)\)", "dda", None),
        ("unified K-branch / reps rows (6 rows; within 3.5 SE + 0.0001 except the K=3 cubic S=20 row)",
         r"(K=3 PditCycle\+PditShift (?:quad|cubic)\s+S=\s*\d+|reps=2 PNOT\+PCNOT\s+quad\s+S=\s*\d+)\s+_filter predicted \[([-\d. ]+)\]  measured \[([-\d. ]+)\]  SE up to ([\d.]+)", "unified", None),
        ("empty-branch exact binomial sum matches measured (6 rows, within 3.5 SE + 0.0001)", r"delta method (-?[\d.]+), exact (-?[\d.]+), measured (-?[\d.]+) \(SE ([\d.]+)\)", "empty", None),
        ("full gradient sd: _filter and REINFORCE about half of _inf",
         r"(_inf \(S=1000\)|_filter \(S=7000\)|REINFORCE \+ leave-one-out baseline \(M=7000\))\s+mean \[[-\d. ]+\]  sd \[([\d. ]+)\]", "fullgrad", None),
        ("chain deterministic Adam prediction (400 and 4000 steps)", r"steps: E = ([\d.]+), theta", [0.75, 0.75], 0.002),
        ("Eq. 44 score term matches exact", r"inner expectation exact : \[([\d.]+)", 0.4451, 0.004),
        ("Eq. 44 auxiliary-sample variance higher", r"sd per sample \[([\d.]+)", 0.833, 0.05),
        ("Eq. 44 downstream: baseline cuts the aux sd", r"auxiliary sample \+ baseline[^\n]*sd per sample \[([\d.]+)", 0.611, 0.05),
        ("Eq. 44 downstream: exact-score baseline sd", r"exact score \+ baseline[^\n]*sd per sample \[([\d.]+)", 0.389, 0.05),
    ],
    "torx_learn.py": [
        ("PISING J=0.3 <ab>", r"J=0\.3 +<ab>=\+?([\d.]+)", 0.0552, 0.001),
        ("RMS N=10000", r"N= 10000  RMS error =\s+([\d.]+)", 3.33, 1.0),
    ],
    "z1_attention.py": [
        ("method B error N=1e4", r"10000\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)", 2.37, 0.6),
        ("attention TV N=100,1000,1e4", r"N=\s*\d+\s+TV = ([\d.]+)", [0.1270, 0.0385, 0.0124], 0.02),
    ],
    "onchip_gate.py": [("lam=4 RMS", r"\n\s+4\s+([\d.]+)", 8.0, 2.5)],
}


def evaluate(desc, out, rx, want, tol):
    """Return (ok, message)."""
    if want is None:                                       # custom checks
        if desc == "N required":
            vals = [int(x.replace(",", "")) for x in re.findall(rx, out)]
            ok = vals == [13, 147, 1323] or all(abs(a - b) / b < 0.1 for a, b in zip(vals, [13, 147, 1323]))
            return ok, f"{vals}"
        if desc.startswith("OU"):
            m = re.search(rx, out); mean, var = float(m.group(1)), float(m.group(2))
            return abs(mean - 0.878) < 0.02 and abs(var - 0.4667) < 0.02, f"{mean}, {var}"
        if desc.startswith("inf per-gate"):
            pairs = re.findall(rx, out)
            ok = all(abs(float(a) - float(b)) <= 0.001 for a, b in pairs) and len(pairs) == 5
            return ok, f"{pairs}"
    if want == "filtpairs":
        pairs = re.findall(rx, out)
        ok = len(pairs) == 3 and all(abs(float(a) - float(b)) / float(a) < 0.05 for a, b in pairs)
        return ok, str(pairs)
    if want == "lawpairs":
        rows = re.findall(rx, out)
        ok = len(rows) == 2 and all(abs(float(m) - float(pred)) / float(pred) < 0.4 for pred, ms in rows for m in ms.split())
        return ok, str(rows)
    if want == "law":
        rows = re.findall(rx, out)
        ok = len(rows) == 10
        for _S, pred, meas, se in rows:
            pr, ms = [float(x) for x in pred.split()], [float(x) for x in meas.split()]
            ok = ok and len(pr) == len(ms) and all(abs(m - p_) <= 3 * float(se) + 0.0003 for p_, m in zip(pr, ms))
        return ok, "%d rows" % len(rows)
    if want == "filtlaw":
        rows = re.findall(rx, out)
        ok = len(rows) == 7
        for pr_, ms_, se_ in rows:
            pr, ms = [float(x) for x in pr_.split()], [float(x) for x in ms_.split()]
            ok = ok and len(pr) == len(ms) and all(abs(a - b) <= 3.5 * float(se_) + 0.0001 for a, b in zip(pr, ms))
        return ok, "%d rows" % len(rows)
    if want == "filtvar":
        m = re.search(rx, out)
        return bool(m) and abs(float(m.group(1)) - float(m.group(2))) / float(m.group(1)) < 0.05, str(m.groups() if m else None)
    if want == "dda":
        m = re.search(rx, out)
        return bool(m) and abs(float(m.group(1)) - float(m.group(2))) <= 0.0012, str(m.groups() if m else None)
    if want == "unified":
        rows = re.findall(rx, out)
        ok = len(rows) == 6
        for label, pr_, ms_, se_ in rows:
            if "cubic" in label and "S= 20" in label:
                continue                                         # second-order expansion strains here (documented)
            pr, ms = [float(x) for x in pr_.split()], [float(x) for x in ms_.split()]
            ok = ok and all(abs(a - b) <= 3.5 * float(se_) + 0.0001 for a, b in zip(pr, ms))
        return ok, "%d rows" % len(rows)
    if want == "empty":
        rows = re.findall(rx, out)
        return len(rows) == 6 and all(abs(float(c_) - float(m_)) <= 3.5 * float(s_) + 0.0001 for d_, c_, m_, s_ in rows), str(rows)
    if want == "fullgrad":
        rows = dict((k_, [float(x) for x in v_.split()]) for k_, v_ in re.findall(rx, out))
        if len(rows) != 3:
            return False, str(rows)
        inf_, fil_, rf_ = rows["_inf (S=1000)"], rows["_filter (S=7000)"], rows["REINFORCE + leave-one-out baseline (M=7000)"]
        return all(f_ < 0.65 * i_ and r_ < 0.65 * i_ for i_, f_, r_ in zip(inf_, fil_, rf_)), str(rows)
    if want == "infpairs":
        rows = re.findall(rx, out)
        return len(rows) == 4 and all(abs(float(a) - float(b)) <= 0.0012 for a, b in rows), str(rows)
    if want == "kbatch":
        rows = re.findall(rx, out)
        if len(rows) != 2:
            return False, str(rows)
        single = [float(x) for x in rows[0].split()]; three = [float(x) for x in rows[1].split()]
        return abs(single[-1] + 0.0010) <= 0.0006 and max(abs(x) for x in three) <= 0.0002, str(rows)
    if want == "ratio":
        rows = re.findall(rx, out)
        vals = [float(x) for r_ in rows for x in r_.split()]
        return len(rows) == 2 and all(0.85 <= v <= 1.25 for v in vals), str(rows)
    if want == "gatedep":
        rows = re.findall(rx, out)
        ok = len(rows) == 3 and all(len(set(r_.split())) > 1 for r_ in rows)
        return ok, str(rows)
    if want == "fixed":
        rows = re.findall(rx, out)
        return len(rows) == 6 and all(abs(float(a) - float(b)) < 0.0025 for a, b in rows), str(rows)
    if want == "k3pairs":
        rows = re.findall(rx, out)
        ok = len(rows) == 2 and all(abs(float(a) - float(b)) / float(a) < 0.1 for p_, m_ in rows for a, b in zip(p_.split(","), m_.split(",")))
        return ok, str(rows)
    if want == "winners":
        w = re.findall(rx, out)
        return w == ["_inf", "_filter", "_filter", "_inf"], str(w)
    if isinstance(want, str) and want.startswith("pairs"):
        tol_ = float(want[5:]); pairs = re.findall(rx, out)
        return len(pairs) == 2 and all(abs(float(a) - float(b)) <= tol_ for a, b in pairs), str(pairs)
    if want == "max<0.008":
        vals = [float(x) for x in re.findall(rx, out)]
        return len(vals) >= 15 and max(vals) < 0.008, f"n={len(vals)} max={max(vals) if vals else None}"
    if isinstance(want, str):
        m = re.search(rx, out)
        return bool(m) and m.group(1).strip().lower() == want.lower(), str(m.group(1) if m else None)
    got = [float(x) for x in re.findall(rx, out)]
    exp = want if isinstance(want, list) else [want]
    if len(got) < len(exp):
        return False, f"matched {len(got)} of {len(exp)}"
    ok = all(abs(g - e) <= tol for g, e in zip(got, exp))
    return ok, f"got {got[:len(exp)]} want {exp} +/- {tol}"


DOCS = ["../gradients.md", "../estimator-math.md", "../explain-code.md", "../../SKILL.md", "../gates.md",
        "../factors-dfg-hybrid.md", "../thrml-vs-torx.md", "../hardware-facts.md"]
NUM = re.compile(r"(?<![\w.])[-+]?\d+\.\d+(?:e-?\d+)?(?![\d])")      # every decimal number, incl. scientific notation
FLOAT = re.compile(r"[-+]?\d+\.\d+(?:e[-+]?\d+)?")
# numbers in the docs that are quoted from the paper/notebooks or derived by hand (arithmetic shown in the doc), not printed by a script
ALLOW_REASONS = {
    "0.0000": "rounded zero; numpy prints 0. or -0.",
    "0.0634": "notebook 06 chromatic-Gibbs TV (quoted, not re-run)",
    "0.4020": "notebook 06 bond-local PISING TV (quoted)", "0.402": "same value, paper p. 13",
    "0.0525": "whitepaper p. 14 IS software TV", "0.0887": "whitepaper p. 14 MH software TV",
    "0.1126": "whitepaper p. 14 MH hardware TV", "0.0622": "whitepaper p. 14 IS hardware TV",
    "0.052": "whitepaper p. 13 chromatic-Gibbs TV (quoted)",
    "0.999995": "historic saturated z1_attention demo (replaced; kept as a warning)",
    "2608.01612": "arXiv identifier of the Torx whitepaper", "2608.01615": "arXiv identifier of the Thermalizers paper",
    "0.999": "clip constant used in the code (np.clip(m, -0.999, 0.999))",
    "3.11": "Python version requirement (3.11+)",
    "2.16": "hand arithmetic 269,568 x 16 / 2 = 2.16 M edges",
    "14.1": "historic saturated demo logit (replaced)", "0.7": "historic saturated demo logit (replaced)",
    "0.43": "logit gap of the 8 evenly spaced logits in [-1.5, 1.5]: 3/7",
    "1.4": "derived SE multiple (difference / SE) stated in the text", "1.8": "derived SE multiple",
    "1.9": "derived SE multiple", "2.5": "derived SE multiple", "3.2": "derived SE multiple",
    # integers (checked since round 3)
    "2592": "p-bit count printed by z1_attention.py (9 x 288); the paragraph cites the cookbook that reuses it",
    "5000": "rounded 'about' value of the printed N_eff 4,975", "16000": "rounded 'about' count of one-hot samples (20,000 x 0.80)",
    "256": "2^8 states of the 8-pbit example (stated arithmetic)", "2027": "year (Z1 early access, blog)",
    "107": "number of runner checks (self-reference)", "215904": "blog figure: Z1 coupling parameters",
}
ALLOW = set(ALLOW_REASONS)



SCRIPT_REF = re.compile(r"cookbook_(\d+)(?:_\w+)?(?:\.py)?|torx_learn|z1_attention|onchip_gate")
SEP = re.compile(r"^[\s,/;]*$")


def _scripts_in(line, names):
    """Script file names (keys of `names`) mentioned on a line; `cookbook_N` matches the script starting `cookbook_N_`."""
    found = set()
    for m in SCRIPT_REF.finditer(line):
        txt = m.group(0)
        for n in names:
            base = n[:-4] if n.endswith(".txt") else n
            if m.group(1) is not None and base.startswith("cookbook_%s_" % m.group(1)):
                found.add(n)
            elif m.group(1) is None and base.startswith(txt):
                found.add(n)
    return found


def _value_matches(tok, printed_value):
    """Signed (if the doc writes a sign) match after rounding to the doc's own precision; scientific notation compares mantissas."""
    bare = tok.lstrip("-").lstrip("+")
    if "e" in bare:
        mant, ex = bare.split("e")
        dec = len(mant.split(".")[1]); scale = 10.0 ** int(ex)
        target = float(tok) / scale; v = printed_value / scale
    else:
        dec = len(bare.split(".")[1]); target = round(float(tok), dec); v = printed_value
    if tok.startswith(("-", "+")):
        return round(v, dec) == round(target, dec)
    return printed_value >= 0 and round(v, dec) == round(target, dec)   # unsigned doc number: printed value must be non-negative


def _source_literals(script):
    """Float literals in a script's source: the doc may quote an INPUT (theta, lr, S, t) that the script sets but does not print."""
    path = os.path.join(HERE, script[:-4] if script.endswith(".txt") else script)
    if not os.path.exists(path):
        return []
    return [float(x) for x in FLOAT.findall(open(path, encoding="utf8").read())]


INPUT_CTX = re.compile(r"(?:(?<![<>!])=|[\[<>]|\b(?:below|above|under|over|threshold)|\b(?:theta\w*|lr|S|t|N|d|M|K|m|q|rho|target|seed|beta|lam|lambda|w|b|bias|range)\s*=?)\s*$")
WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9']+|[A-Za-z]\([^)\s]{1,6}\)")
ANYNUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:e-?\d+)?(?![\d])")


def _printed_lines(names, outputs):
    out = []
    for n in names:
        for pline in outputs[n].split("\n"):
            vals = [float(x) for x in ANYNUM.findall(pline)]
            if vals:
                out.append((pline.lower(), vals))
    return out


def _lone_ok(tok, scripts, outputs, ctx):
    if ctx == "LENIENT":                                               # shown arithmetic in inline code: printed anywhere or an input
        names = list(scripts) if scripts else list(outputs)
        return any(_value_matches(tok, v) for _l, vals in _printed_lines(names, outputs) for v in vals) or             any(_value_matches(tok, v) for n in names for v in _source_literals(n))
    """A lone number must be printed on a line that ALSO carries context: another number from the same doc line, or the word that
    precedes the number in the doc. A source literal (an input the script sets) counts only when the doc writes it as an input
    (`theta=`, `lr`, `S=`, `target` ...). This stops a wrong result passing because the same value is printed elsewhere."""
    names = list(scripts) if scripts else list(outputs)
    prev, sibs, label, post, span = ctx if ctx else ("", [], "", "", "")
    if INPUT_CTX.search(prev[-20:]) and any(_value_matches(tok, v) for n in names for v in _source_literals(n)):
        return True
    words = [w.lower() for w in WORD.findall(prev[-40:]) if w.lower() not in STOP][-3:]
    words += [w.lower() for w in WORD.findall(label) if w.lower() not in STOP]
    words += [w.lower() for w in WORD.findall(re.split(r"\d", post, maxsplit=1)[0][:30]) if w.lower() not in STOP][:2]   # words right after
    words += [w.lower() for w in WORD.findall(span) if w.lower() not in STOP]                               # enclosing inline code
    words = [w for w in words if len(w) >= 2]
    near = [x for x in re.findall(r"(?<![\w.])\d+(?![\w.])", prev[-16:] + " " + post[:16])]          # small ints next to it ("at lam = 1")
    sibs = list(sibs) + near
    words = sorted(set(words))
    sibs = [x for x in sibs if x != tok]
    if not words and not sibs:
        return any(_value_matches(tok, v) for _l, vals in _printed_lines(names, outputs) for v in vals)

    def wscore(low):
        pw = [x for x in WORD.findall(low) if len(x) >= 4]
        return sum(1 for w in words if w in low or any(x in w for x in pw))

    def score(low, vals):
        pw = [x for x in WORD.findall(low) if len(x) >= 4]
        sw = sum(1 for w in words if w in low or any(x in w for x in pw))
        ss = sum(1 for sb in set(sibs) if any((_value_matches(sb, v) if "." in sb else float(sb) == v) for v in vals))
        return sw + 2 * ss
    scored = [(score(low, vals), vals) for low, vals in _printed_lines(names, outputs)]
    # bar = best WORD-only match over all printed lines; a candidate line also earns credit for the doc's neighbouring numbers
    best = max((wscore(low) for low, _v in _printed_lines(names, outputs)), default=0) if words else 1
    if best == 0 and not any(sc_ > 0 for sc_, _v in scored):     # no printed line shares any context: value check only
        return any(_value_matches(tok, v) for _sc, vals in scored for v in vals)
    # the value must sit on (one of) the printed lines that match the doc context BEST, not merely on some line sharing one word
    return any(sc_ >= best - SLACK and sc_ > 0 and any(_value_matches(tok, v) for v in vals) for sc_, vals in scored)


WIN = 60
SLACK = int(os.environ.get("CHK_SLACK", "99"))   # context sanity only; the lock (doclock.py) is the strict guard


STOP = {"the", "and", "of", "at", "in", "to", "is", "a", "an", "vs", "with", "for", "on", "by", "from", "about", "gives", "give",
        "be", "are", "it", "as", "or", "than", "then", "so", "its", "per", "each", "all", "only"}


def _group_ok(tokens, scripts, outputs, ctx=None, strict=True):
    """Tokens of a comma/slash separated group must be covered by chunks of >= 2 CONSECUTIVE tokens, each chunk found on ONE printed
    line of a bound script (so a per-gate triple or a predicted/measured pair cannot be altered in one component); a lone token
    (group of one) must pass `_lone_ok` (printed together with doc context)."""
    toks = [t for t in tokens if t.lstrip("-").lstrip("+") not in ALLOW]
    if not toks:
        return True
    names = list(scripts) if scripts else list(outputs)
    n = len(toks)
    if n == 1:
        return _lone_ok(toks[0], scripts, outputs, ctx)
    lines = [vals for _low, vals in _printed_lines(names, outputs)]
    if ctx and ctx != "LENIENT" and INPUT_CTX.search(ctx[0][-20:]):
        lines += [l_ for l_ in (_source_literals(nm) for nm in names) if l_]
    # chunk_ok[i][j]: tokens i..j-1 all found on one printed line
    def in_order(ts, vals):                                       # tokens appear on the line in the doc's order
        k = 0
        for t in ts:
            while k < len(vals) and not _value_matches(t, vals[k]):
                k += 1
            if k == len(vals):
                return False
            k += 1
        return True

    def chunk(i, j):
        return any(in_order(toks[i:j], vals) for vals in lines)
    ok = [False] * (n + 1); ok[0] = True
    for j in range(2, n + 1):
        ok[j] = any(ok[i] and j - i >= 2 and chunk(i, j) for i in range(0, j - 1))
    return ok[n]


EXCLUDED = [0]
ARITH = [0]
# integers with >= 3 digits (or comma grouped, 4,975) in prose/tables: repeats, S, N_eff, counts. Not part of a decimal or an identifier.
INT = re.compile(r"(?<![\w.,\-])(?:\d{1,3}(?:,\d{3})+|\d{3,})(?![\w]|\.\d|,\d)")
INT_LIT = re.compile(r"(?<![\d.])\d[\d_,]*(?![\d_,]*\.\d)")
# result numbers that follow a key word: the printed line must carry the SAME key in front of the value ("p = 0.96", "at step 13"),
# so a value printed elsewhere in a dense output cannot stand in for it
KEYED = re.compile(r"(?<![\w.])(p\s*=|t\s*=|at step)\s*([-+]?\d+(?:\.\d+)?)(?!\d|\.\d)")


def _ints(text):
    """Integers in a printed output or a source file, plus printed floats rounded to an integer (the doc's own precision)."""
    out = {int(x.replace(",", "").replace("_", "")) for x in INT_LIT.findall(text) if x.replace(",", "").replace("_", "").strip(",").isdigit()}
    return out | {round(float(x)) for x in FLOAT.findall(text)}


def _int_ok(tok, scripts, outputs):
    if tok.replace(",", "") in ALLOW:
        return True
    names = list(scripts) if scripts else list(outputs)
    v = int(tok.replace(",", ""))
    for n in names:
        src = os.path.join(HERE, n[:-4] if n.endswith(".txt") else n)
        pool = _ints(outputs[n]) | (_ints(open(src, encoding="utf8").read()) if os.path.exists(src) else set())
        if v in pool:
            return True
    return False


def _keyed_ok(key, tok, scripts, outputs):
    names = list(scripts) if scripts else list(outputs)
    for n in names:
        for k_, v_ in KEYED.findall(outputs[n]):
            if k_.replace(" ", "") == key.replace(" ", "") and ("." in tok) == ("." in v_) and (_value_matches(tok, float(v_)) if "." in tok else int(tok) == int(v_)):
                return True
    return False


def doc_extra_entries(outputs):
    """Integer and keyed entries per doc line (same binding and exclusions as doc_entries): ('int', tok) or ('key', key, tok)."""
    out = []
    for rel in DOCS:
        path = os.path.normpath(os.path.join(HERE, rel))
        current, in_code = set(), False
        for ln, line in enumerate(open(path, encoding="utf8").read().split("\n"), 1):
            if line.lstrip().startswith("```"):
                in_code = not in_code
                continue
            if not in_code:
                mentioned = _scripts_in(line, outputs.keys())
                if mentioned:
                    current = mentioned
            if in_code or line.lstrip().startswith("#"):
                continue
            plain = re.sub(r"`[^`]*`", " ", line)                   # inline code (identifiers, snippets) is not a result
            plain = re.sub(r"\]\([^)]*\)", "]", plain)              # link targets
            for m in INT.finditer(plain):
                out.append((os.path.basename(path), ln, ("int", m.group(0)), set(current)))
            for m in KEYED.finditer(plain):
                out.append((os.path.basename(path), ln, ("key", m.group(1), m.group(2)), set(current)))
    return out


def _extra_ok(item, sc, outputs):
    return _int_ok(item[1], sc, outputs) if item[0] == "int" else _keyed_ok(item[1], item[2], sc, outputs)


def doc_entries(outputs):
    """Groups of numbers per doc line with the scripts bound to them: the binding is the most recent line (same doc) that names a
    script, so a number must be printed by the script its paragraph cites."""
    entries = []
    for rel in DOCS:
        path = os.path.normpath(os.path.join(HERE, rel))
        current = set()
        in_code = False
        for ln, line in enumerate(open(path, encoding="utf8").read().split("\n"), 1):
            if line.lstrip().startswith("```"):
                in_code = not in_code
                continue
            if not in_code:
                mentioned = _scripts_in(line, outputs.keys())   # headings also bind the scripts for the section below
                if mentioned:
                    current = mentioned
            if in_code or line.lstrip().startswith("#"):
                EXCLUDED[0] += len(NUM.findall(line))           # code and headings are not checked (counted, reported)
                continue
            ms = list(NUM.finditer(line))
            # inside an inline code span only the final result (the number after the span's last "=") is a claim; the other numbers
            # are shown arithmetic, inputs or formula constants that the reader can verify by eye (counted in ARITH, reported)
            spans = [(m_.start(), m_.end(), m_.group(0)) for m_ in re.finditer(r"`[^`]*`", line)]
            keep = []
            for m in ms:
                sp = next((x for x in spans if x[0] < m.start() < x[1]), None)
                if sp is None:
                    keep.append(m); continue
                last_eq = sp[2].rfind("=")
                tail = sp[2][last_eq + 1:] if last_eq >= 0 else ""
                if last_eq >= 0 and m.start() > sp[0] + last_eq and re.fullmatch(r"\s*[-+]?[\d.e-]+\s*`?", tail):
                    keep.append(m)
                else:
                    ARITH[0] += 1
                    entries.append((os.path.basename(path), ln, [m.group(0)], set(current), "LENIENT"))
            ms = keep
            groups, cur = [], []
            for m in ms:
                if cur and SEP.match(line[cur[-1].end():m.start()]):
                    cur.append(m)
                else:
                    if cur:
                        groups.append(cur)
                    cur = [m]
            if cur:
                groups.append(cur)
            all_pos = [(m.start(), m.group(0)) for m in ms] + [(m_.start(), m_.group(0)) for m_ in re.finditer(r"(?<![\w.,])\d{2,}(?![\w]|\.\d)", line)]
            label = line.split("|")[1] if line.lstrip().startswith("|") and line.count("|") > 2 else ""
            for g in groups:
                cell_start = line.rfind("|", 0, g[0].start()) + 1
                prev = line[cell_start:g[0].start()]
                cell_end = line.find("|", g[-1].end()); cell_end = len(line) if cell_end < 0 else cell_end
                post = line[g[-1].end():cell_end]
                sp = next((x for x in spans if x[0] < g[0].start() < x[1]), None)
                lo_, hi_ = g[0].start() - WIN, g[-1].end() + WIN    # siblings: numbers in the same clause window, not the whole line
                cell_l = line.rfind("|", 0, g[0].start()); cell_r = line.find("|", g[-1].end())
                if line.lstrip().startswith("|"):
                    lo_, hi_ = max(lo_, cell_l), (hi_ if cell_r < 0 else min(hi_, cell_r))
                    sib = [t for q, t in all_pos if lo_ <= q < hi_] + re.findall(r"(?<![\w.,])\d{2,}(?![\w]|\.\d)", label)
                else:
                    sib = [t for q, t in all_pos if lo_ <= q < hi_]
                entries.append((os.path.basename(path), ln, [m.group(0) for m in g], set(current), (prev, sib, label, post, sp[2] if sp else "")))
    return entries


def doc_numbers_check(outputs):
    """Every number with >= 3 decimals in a table row or bullet must equal (signed if the doc writes a sign, rounded to the doc's own
    decimals) a float printed by the script that paragraph cites, grouped numbers on one printed line, or be in ALLOW."""
    return [(f, ln, " ".join(toks)) for f, ln, toks, sc, cx in doc_entries(outputs) if not _group_ok(toks, sc, outputs, cx)]


def mutation_catch_rate(outputs, k_units=(3, 10, 30)):
    """Perturb ONE number of each passing group by k units of its last decimal; report how many perturbations the check catches."""
    entries = [e for e in doc_entries(outputs) if _group_ok(e[2], e[3], outputs, e[4])]
    rates = {}
    for k in k_units:
        caught = total = 0
        lone_caught = lone_total = 0
        for _f, _ln, toks, sc, cx in entries:
            for i, tok in enumerate(toks):
                if tok.lstrip("-").lstrip("+") in ALLOW:
                    continue
                dec = len(tok.lstrip("-").lstrip("+").split(".")[1]); v = float(tok)
                mutated = ("%+.*f" % (dec, v + k * 10 ** (-dec))) if tok.startswith(("-", "+")) else ("%.*f" % (dec, v + k * 10 ** (-dec)))
                total += 1
                c = (not _group_ok(toks[:i] + [mutated] + toks[i + 1:], sc, outputs, cx))
                caught += c
                if len(toks) == 1:
                    lone_total += 1; lone_caught += c
        rates[k] = (caught, total, lone_caught, lone_total)
    return rates


def substitution_catch_rate(outputs, per_token=8):
    """Realistic error model: replace ONE doc number by ANOTHER value printed by the same bound script(s) or set in its source
    (a neighbouring row, the target, an input), formatted at the doc's precision; report how often the check catches it."""
    import random
    rnd = random.Random(0)
    entries = [e for e in doc_entries(outputs) if e[4] != "LENIENT" and _group_ok(e[2], e[3], outputs, e[4])]
    caught = total = 0
    for _f, _ln, toks, sc, cx in entries:
        names = list(sc) if sc else list(outputs)
        pool = sorted({v for n in names for v in [float(x) for x in FLOAT.findall(outputs[n])] + _source_literals(n)})
        for i, tok in enumerate(toks):
            bare = tok.lstrip("-+")
            if bare in ALLOW or "e" in bare:
                continue
            dec = len(bare.split(".")[1])
            cands = sorted({"%.*f" % (dec, abs(v)) for v in pool} - {"%.*f" % (dec, abs(float(tok)))} - {"%.*f" % (dec, 0.0)})
            for c in rnd.sample(cands, min(per_token, len(cands))):
                sub = ("-" if tok.startswith("-") else "+" if tok.startswith("+") else "") + c
                total += 1
                caught += not _group_ok(toks[:i] + [sub] + toks[i + 1:], sc, outputs, cx)
    return caught, total


PDF = os.environ.get("TORX_WP_PDF") or os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..", "2608.01612v2.pdf"))
WP_TAG = re.compile(r"\[WP ([^\]]*)\]")
WP_UNIT = re.compile(r"\s*-?\s*(%|ns|ms|us|\u00b5s|aJ|fJ|MHz|GHz|mm|W|bit|sites?|samples|neighbours|neighbors)(?![A-Za-z])")


def wp_check():
    """[WP pN] facts: every number in the tagged table cell or sentence (1-2 digit integers included) must occur on the cited
    whitepaper page(s) of the local PDF (pypdf text, whitespace-normalised). Returns (checked, misses) or None without pypdf/PDF."""
    try:
        import pypdf
    except ImportError:
        return None
    if not os.path.exists(PDF):
        return None
    pages = [pg.extract_text() or "" for pg in pypdf.PdfReader(PDF).pages]
    norm = lambda t: re.sub(r"\s+", " ", t.replace("−", "-").replace("-\n", "").replace(",", ""))
    checked, misses = 0, []
    for rel in DOCS:
        path = os.path.normpath(os.path.join(HERE, rel)); in_code = False
        for ln, line in enumerate(open(path, encoding="utf8").read().split("\n"), 1):
            if line.lstrip().startswith("```"):
                in_code = not in_code
            if in_code:
                continue
            for m in WP_TAG.finditer(line):
                nums = [int(x) for x in re.findall(r"p(\d+)", m.group(1))]
                rng = re.search(r"p(\d+)-(\d+)", m.group(1))
                if rng:
                    nums += list(range(int(rng.group(1)), int(rng.group(2)) + 1))
                if not nums:
                    continue
                text = norm(" ".join(pages[n - 1] for n in set(nums) if 0 < n <= len(pages)))
                if line.lstrip().startswith("|"):                 # table: the fact cell(s) before the tag cell
                    seg = line[:m.start()].rsplit("|", 2)[-2] if line[:m.start()].count("|") >= 2 else line[:m.start()]
                else:                                             # prose: the sentence that ends in the tag
                    seg = re.split(r"(?<=[.;])\s", line[:m.start()])[-1]
                seg = re.sub(r"arXiv:\S+", " ", seg)
                seg = seg.replace(",", "")
                bare_text = re.sub(r"\[\d+(?:\s*[,\u2013-]\s*\d+)*\]", " ", text)   # drop citation markers like [9] or [8, 9]
                for mm in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?(?![\d])", seg):
                    x = mm.group(0); checked += 1
                    after, before = seg[mm.end():mm.end() + 14], seg[max(0, mm.start() - 8):mm.start()]
                    unit = WP_UNIT.match(after)
                    if unit:                                      # the number must appear WITH its unit: 5%, 100 ns, 8-bit
                        u = unit.group(1)
                        u_rx = {"neighbours": "neighbou?rs?", "neighbors": "neighbou?rs?"}.get(u, re.escape(u))
                        rx = r"(?<![\d.])" + re.escape(x) + r"\s*-?\s*" + u_rx
                        ok = re.search(rx, text) is not None
                    elif before.endswith("degree-"):
                        ok = re.search(r"degree-\s*" + re.escape(x) + r"(?!\d)", text) is not None
                    else:                                         # bare number: not a citation marker, not just the page number
                        hits = len(re.findall(r"(?<![\d.])" + re.escape(x) + r"(?![\d]|\.\d)", bare_text))
                        ok = hits >= (2 if x.isdigit() and int(x) in nums else 1)
                    if not ok:
                        misses.append((os.path.basename(path), ln, x + (unit.group(0).rstrip() if unit else ""), m.group(0)))
    return checked, misses


LOCK_SELFTEST = [
    # (name, old text, new text, expect): edits applied in memory to gradients.md, never written to disk
    ("relabel one repeated table row, numbers unchanged", "| 20 | `_filter` | 0.8434 | 0.8421 |",
     "| 20 | `_filter` (single gate) | 0.8434 | 0.8421 |", "clean"),
    ("relabel the S=50 `_single` row of the same block", "| 50 | `_single` | 0.7794 | 0.7800 |", "| 50 | `_single` (S=50 row) | 0.7794 | 0.7800 |", "clean"),
    ("relabel all five rows of the repeated block at once", ["| 20 | `_inf` |", "| 20 | `_single` |", "| 20 | `_filter` |", "| 50 | `_inf` |", "| 50 | `_single` |"],
     ["| 20 | `_inf` (a) |", "| 20 | `_single` (b) |", "| 20 | `_filter` (c) |", "| 50 | `_inf` (d) |", "| 50 | `_single` (e) |"], "clean"),
    ("insert an unrelated paragraph without numbers above the table", "| 20 | `_inf` |", "Rows below are per S.\n\n| 20 | `_inf` |", "clean"),
    ("wrong value in a table row (target in place of the measurement)", "| 20 | `_filter` | 0.8434 | 0.8421 |",
     "| 20 | `_filter` | 0.8000 | 0.8421 |", "refused"),
    ("reworded table row with a wrong value", "| 20 | `_filter` | 0.8434 | 0.8421 |",
     "| 20 | `_filter` (fixed point, loss vs target) | 0.8000 | 0.8421 |", "refused"),
    ("fabricated row that was never run (S=100)", "| 50 | `_filter` | 0.8176 | 0.8163 |",
     "| 50 | `_filter` | 0.8176 | 0.8163 |\n| 100 | `_filter` | 0.8434 | 0.8421 |", "refused"),
    ("row labels swapped, values unchanged", ["| 20 | `_filter` | 0.8434", "| 50 | `_filter` | 0.8176"],
     ["| 50 | `_filter` | 0.8434", "| 20 | `_filter` | 0.8176"], "refused"),
    ("reworded prose with a wrong value", "at S=20 the measured fixed points are 0.8434 (`_filter`)",
     "at S=20 the `_filter` fixed point sits on the target, 0.8000 (`_filter`)", "refused"),
]


# Frozen fixture for the lock self-test: a snapshot of the fixed-point table in gradients.md, so editing the real docs never
# breaks the self-test. Its numbers are printed by cookbook_13 part 2; if that output changes the self-test is skipped with a
# warning (update the snapshot), not failed.
LOCK_FIXTURE = "\n".join([
    "**Fixed points (`cookbook_13_bias_theory.py` part 2; single `PNOT`, loss `(E_hat - 0.8)^2`).**",
    "",
    "| S | method | measured E* | predicted |",
    "|---|---|---|---|",
    "| 20 | `_inf` | 0.8002 | 0.8000 |",
    "| 20 | `_single` | 0.7509 | 0.7500 |",
    "| 20 | `_filter` | 0.8434 | 0.8421 |",
    "| 50 | `_inf` | 0.8006 | 0.8000 |",
    "| 50 | `_single` | 0.7794 | 0.7800 |",
    "| 50 | `_filter` | 0.8176 | 0.8163 |",
    "",
    "For a single gate from a fixed input: at S=20 the measured fixed points are 0.8434 (`_filter`) and 0.7509 (`_single`) around the target 0.8.",
])


def lock_selftest(outs, items_fn):
    """Edit the frozen fixture in memory, relock against a mini-lock built from it (nothing saved), check. A cosmetic edit must
    give 0 refused, 0 removed and a clean check; a wrong value must be refused at relock and keep failing afterwards.
    Returns None (with a reason) when the fixture is no longer printed by cookbook_13, so the caller warns instead of failing."""
    import doclock
    only = lambda its: [it for it in its if it[0] == "gradients.md"]
    base_items = only(items_fn({"gradients.md": LOCK_FIXTURE}))
    lock, rep0 = doclock.relock(base_items, outs, HERE, [])
    unbound = [it[2] for it in rep0["unbound"] if it[2].lstrip("+-") not in ALLOW]
    if rep0["refused"] or unbound or doclock.check(base_items, lock, outs, HERE, ALLOW):
        return None, "fixture numbers no longer bind to cookbook_13 output (update LOCK_FIXTURE): %s" % (unbound or rep0["refused"][:2])
    results = []
    for name, a, b, expect in LOCK_SELFTEST:
        aa, bb = (a, b) if isinstance(a, list) else ([a], [b])
        text = LOCK_FIXTURE
        for x, y in zip(aa, bb):
            assert x in text, "self-test anchor missing from LOCK_FIXTURE: " + x
            text = text.replace(x, y, 1)
        items = only(items_fn({"gradients.md": text}))
        new, rep = doclock.relock(items, outs, HERE, lock)
        after = doclock.check(items, new, outs, HERE, ALLOW) + [(e["file"], e["tok"], "stale") for e in doclock.stale(items, new)]
        nref = len(rep["refused"]) + len([u for u in rep["unbound"] if u[2].lstrip("+-") not in ALLOW])
        nrem = len(rep["removed"])
        ok = (nref == 0 and nrem == 0 and not after) if expect == "clean" else (nref >= 1 and len(after) >= 1)
        results.append((name, ok, "%d refused or unbound, %d removed, %d failing after relock" % (nref, nrem, len(after))))
    return results, ""


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    save = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--save=")), None)
    docs_only = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--docs-only=")), None)
    quick = "--quick" in sys.argv
    only = set(args) | (QUICK if quick else set())
    fails, outputs = 0, {}
    if docs_only:
        for fn in os.listdir(docs_only):
            outputs[fn] = open(os.path.join(docs_only, fn), encoding="utf8").read()
    else:
        with tempfile.TemporaryDirectory() as tmp:
            for script, checks in CHECKS.items():
                if only and script not in only:
                    continue
                r = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=tmp, env=ENV,
                                   capture_output=True, text=True, encoding="utf-8")
                if r.returncode != 0:
                    print(f"FAIL {script}: exit {r.returncode}\n{r.stderr[-400:]}"); fails += 1; continue
                outputs[script] = r.stdout
                if save:
                    os.makedirs(save, exist_ok=True)
                    open(os.path.join(save, script + ".txt"), "w", encoding="utf8").write(r.stdout)
                for desc, rx, want, tol in checks:
                    ok, msg = evaluate(desc, r.stdout, rx, want, tol)
                    print(f"{'ok  ' if ok else 'FAIL'} {script}: {desc}: {msg}")
                    fails += (not ok)
    if not only:
        EXCLUDED[0] = 0; ARITH[0] = 0
        miss = doc_numbers_check(outputs)
        print(f"DOCS {len(DOCS)} files checked (all prose, tables and bullets; {EXCLUDED[0]} numbers inside code blocks or headings excluded; "
              f"{len(ALLOW)} ALLOW entries, each with a reason in ALLOW_REASONS; {ARITH[0]} shown-arithmetic numbers inside inline code checked leniently)")
        for f, ln, tok in miss:
            print(f"DOC  {f}:{ln}: number {tok} not found in any script output (add a check or ALLOW with a reason)")
        fails += len(miss)
        extra = doc_extra_entries(outputs)
        bad = [(f, ln, it) for f, ln, it, sc in extra if not _extra_ok(it, sc, outputs)]
        n_int = sum(it[0] == "int" for _f, _l, it, _s in extra)
        print(f"INTS {n_int} integers (>= 3 digits or comma grouped) and {len(extra) - n_int} keyed values (p =, t =, at step) checked")
        for f, ln, it in bad:
            print(f"DOC  {f}:{ln}: {' '.join(it)} not printed by the bound script (keyed values need the same key on the printed line)")
        fails += len(bad)
        if outputs and not bad:
            good = [(it, sc) for _f, _l, it, sc in extra if not (it[0] == "int" and it[1].replace(",", "") in ALLOW)]
            def bump(it, k):
                if it[0] == "int":
                    v = int(it[1].replace(",", "")); return ("int", "{:,}".format(v + k * max(1, v // 100)) if "," in it[1] else str(v + k * max(1, v // 100)))
                tok = it[2]
                if "." not in tok:
                    return ("key", it[1], str(int(tok) + k))
                dec = len(tok.lstrip("-+").split(".")[1]); return ("key", it[1], "%.*f" % (dec, float(tok) + k * 10 ** (-dec)))
            for k in (1, 3, 10):
                c = sum(not _extra_ok(bump(it, k), sc, outputs) for it, sc in good)
                print(f"MUT  integers/keyed values perturbed by {k} (integers by {k} x max(1, 1%)): caught {c}/{len(good)} ({100 * c / max(len(good), 1):.0f}%)")
        wp = wp_check()
        if wp is None:
            if "--no-wp" in sys.argv:
                print("WP   skipped on request (--no-wp)")
            else:
                print(f"FAIL WP check could not run: pypdf or the whitepaper PDF not found at {PDF} (set TORX_WP_PDF, or pass --no-wp to skip knowingly)")
                fails += 1
        else:
            print(f"WP   {wp[0]} numbers in [WP pN]-tagged facts checked against the cited whitepaper pages")
            for f, ln, x, tag in wp[1]:
                print(f"DOC  {f}:{ln}: {x} not on the page(s) cited by {tag}")
            fails += len(wp[1])
        import doclock
        outs = {(k[:-4] if k.endswith(".txt") else k): v for k, v in outputs.items()}
        items = doclock.doc_items(DOCS, HERE, list(outs), _scripts_in)
        lock_path = os.path.join(HERE, "doc_numbers.lock.json")
        relock_dir = "--relock" in sys.argv
        if relock_dir:
            force = [x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--force-relock=")]
            new, rep = doclock.relock(items, outs, HERE, doclock.load(lock_path), force)

            def show(e):
                return "%s:%s '%s'" % (e["script"], e["pos"], e.get("template", "source literal")[:60])
            for e in rep["added"]:
                print(f"LOCK added    {e['file']}: {e['tok']} -> {show(e)} (score {e.get('score')})")
            for o, e in rep["rebound"]:
                print(f"LOCK rebound  {e['file']}: {o['tok']} @ {show(o)} (score {o.get('score')}) -> {e['tok']} @ {show(e)} (score {e.get('score')})")
            for o, tok, why in rep["refused"]:
                print(f"LOCK REFUSED  {o['file'] if o else ''}: {o['tok'] if o else ''} -> {tok}: {why} (fix the doc, or --force-relock=FILE:TOKEN after checking)")
            for e in rep["removed"]:
                print(f"LOCK removed  {e['file']}: {e['tok']} (was {show(e)})")
            for it in rep["unbound"]:
                if it[2].lstrip("+-").replace(",", "") not in ALLOW:
                    print(f"LOCK unbound  {it[0]}: {it[2]} (no printed value or source literal; ALLOW it with a reason or fix the doc)")
            for e, n_top in rep["ambiguous"]:
                if "--verbose" in sys.argv or any(e is x for x in rep["added"]) or any(e is x for _o, x in rep["rebound"]):
                    print(f"LOCK ambiguous {e['file']}: {e['tok']} -> {show(e)} ({n_top} equally good lines)")
            print(f"LOCK relocked {len(new)}: {len(rep['kept'])} kept, {len(rep['added'])} added, {len(rep['rebound'])} rebound, "
                  f"{len(rep['removed'])} removed, {len(rep['refused'])} REFUSED, {len(rep['ambiguous'])} ambiguous (--verbose lists all)")
            doclock.save(lock_path, new)
            fails += len(rep["refused"])
        lock = doclock.load(lock_path)
        lfails = doclock.check(items, lock, outs, HERE, ALLOW)
        if "cookbook_13_bias_theory.py" in outs:
            st, why = lock_selftest(outs, lambda texts: doclock.doc_items(DOCS, HERE, list(outs), _scripts_in, texts))
            if st is None:
                print(f"WARN lock self-test skipped: {why}")
            for name, ok, msg in st or []:
                print(f"{'ok  ' if ok else 'FAIL'} lock self-test: {name}: {msg}")
                fails += (not ok)
        for e in doclock.stale(items, lock):
            lfails.append((e["file"], e["tok"], "locked doc number no longer in the doc (deleted or rewritten: run --relock and review)"))
        print(f"LOCK {len(items)} doc numbers (decimals, integers >= 3 digits, 'at step N'), {len(lock)} pinned to one printed position or source literal")
        for f, tok, why in lfails:
            print(f"DOC  {f}: {tok}: {why}")
        fails += len(lfails)
        if outputs:
            lc_, lt_ = doclock.substitution_rate(items, lock, outs, HERE)
            print(f"SUB  replacing each doc number by another value printed or set by its script (realistic wrong-value edit): caught {lc_}/{lt_} ({100 * lc_ / max(lt_, 1):.0f}%) by the lock")
            sc_, st_ = substitution_catch_rate(outputs)
            print(f"SUB  same test against the context check alone (no lock): caught {sc_}/{st_} ({100 * sc_ / max(st_, 1):.0f}%)")
            rates = mutation_catch_rate(outputs)
            for k, (c, t, lc, lt) in rates.items():
                print(f"MUT  perturbing each doc number by {k} units of its last decimal: caught {c}/{t} ({100 * c / max(t, 1):.0f}%); lone numbers {lc}/{lt}")
            if rates[30][1] and rates[30][0] / rates[30][1] < 0.85:
                print("FAIL mutation catch rate at k=30 below 85%"); fails += 1
    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
