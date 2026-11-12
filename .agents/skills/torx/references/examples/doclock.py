"""Doc-number lock: pins every number in the English reference docs to ONE position in a script's printed output (or source).

Why: a value check ("is this number printed somewhere by the cited script?") passes a wrong number that happens to be printed
elsewhere, e.g. the target 0.8 written in place of a measured fixed point 0.8434. The lock records, for each doc number, the
printed line (by its text with numbers masked, plus which occurrence) and the index of the number on that line. A later check
reads the value at exactly that position, so any substitution fails, and so does a script whose output drifts.

Workflow: edit a doc number or a script -> `check_examples.py --relock[=DIR]` -> review the printed diff (added / changed /
removed / ambiguous bindings) -> commit the new `doc_numbers.lock.json`. Normal runs never rewrite the lock.
"""
import json
import os
import re

ANY = re.compile(r"(?<![\d.,])[-+]?(?:\d{1,3}(?:,\d{3})+(?![\d.])|\d+)(?:\.\d+)?(?:e[-+]?\d+)?(?![\d])")
DEC = re.compile(r"(?<![\w.])[-+]?\d+\.\d+(?:e-?\d+)?(?![\d])")
INTD = re.compile(r"(?<![\w.,\-])(?:\d{1,3}(?:,\d{3})+|\d{3,})(?![\w]|\.\d|,\d)")
STEP = re.compile(r"(?<![\w.])at step\s*(\d+)(?!\d|\.\d)")
SRCNUM = re.compile(r"(?<![\w.])-?\d[\d_]*(?:\.\d+)?(?:e[-+]?\d+)?(?![\w.])")
WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9']+")
STOP = {"the", "and", "of", "at", "in", "to", "is", "an", "vs", "with", "for", "on", "by", "from", "about", "gives", "give",
        "be", "are", "it", "as", "or", "than", "then", "so", "its", "per", "each", "all", "only", "was", "were"}


def mask(text):
    return re.sub(r"\s+", " ", ANY.sub("#", text)).strip()


def _num(tok):
    return float(tok.replace(",", "").replace("_", ""))


def matches(tok, v):
    """Doc token vs printed value: decimals rounded to the doc's precision (signed if the doc writes a sign); integers exact,
    or a printed float that rounds to it."""
    bare = tok.lstrip("+-").replace(",", "")
    if "e" in bare:
        mant, ex = bare.split("e"); dec = len(mant.split(".")[1]) if "." in mant else 0
        s = 10.0 ** int(ex)
        return round(_num(tok) / s, dec) == round((v if tok[0] in "+-" else abs(v)) / s, dec) and (tok[0] in "+-" or v >= 0)
    dec = len(bare.split(".")[1]) if "." in bare else 0
    if tok[0] in "+-":
        return round(v, dec) == round(_num(tok), dec)
    return v >= 0 and round(v, dec) == round(_num(tok), dec)


def index_output(text):
    """[(template, occurrence, [number tokens])] for every printed line that carries numbers."""
    seen, out = {}, []
    for line in text.split("\n"):
        toks = ANY.findall(line)
        if not toks:
            continue
        t = mask(line); k = seen.get(t, 0); seen[t] = k + 1
        out.append((t, k, toks, line))
    return out


def source_numbers(path):
    if not os.path.exists(path):
        return []
    return SRCNUM.findall(open(path, encoding="utf8").read())


def doc_items(docs, here, script_names, scripts_in, texts=None):
    """Every checkable number in the docs: (file, key, token, bound scripts, context). Code blocks and headings are skipped.
    key = masked line # ordinal of that numbered line in its file # index of the number on the line. Keys are only an
    identity inside one run; old and new docs are matched by aligning their line sequences (`match`), not by key equality.
    `texts` ({file name: text}) replaces file contents (self-tests)."""
    items = []
    for rel in docs:
        path = os.path.normpath(os.path.join(here, rel)); fname = os.path.basename(path)
        current, in_code, ordn = set(), False, 0
        text = texts[fname] if texts and fname in texts else open(path, encoding="utf8").read()
        for line in text.split("\n"):
            if line.lstrip().startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            m_ = scripts_in(line, script_names)
            if m_:
                current = m_
            if line.lstrip().startswith("#"):
                continue
            found = {}
            for rx in (DEC, INTD):
                for m in rx.finditer(line):
                    found.setdefault(m.start(), m.group(0))
            for m in STEP.finditer(line):
                found.setdefault(m.start(1), m.group(1))
            if not found:
                continue
            t = mask(line); occ = ordn; ordn += 1
            label = line.split("|")[1] if line.lstrip().startswith("|") and line.count("|") > 2 else ""
            pos = sorted(found)
            for k, p in enumerate(pos):
                tok = found[p]
                lo = line.rfind("|", 0, p) + 1 if line.lstrip().startswith("|") else 0
                hi = line.find("|", p); hi = len(line) if hi < 0 else hi
                prev, post = line[max(lo, p - 50):p], line[p + len(tok):min(hi, p + len(tok) + 40)]
                if line.lstrip().startswith("|"):
                    # a table row is one record: every other number in the row (small integers included) and the words of
                    # its non-numeric cells are context, and the numbers of the first cell (the row label, e.g. S) must be
                    # printed on the bound line
                    cells = line.strip().strip("|").split("|")
                    small = [x for c in cells for x in re.findall(r"(?<![\w.])\d+(?![\w.])", c)]
                    sibs = [found[q] for q in pos if q != p] + small
                    lab = cells[0].strip()                   # only a plain integer label (S, steps, N) is enforced
                    label_nums = [lab] if len(cells) > 2 and re.fullmatch(r"[-+]?\d+", lab) else []
                    label = " ".join(c for c in cells if not ANY.search(c))
                else:
                    sibs = [found[q] for q in pos if q != p and lo <= q < hi and abs(q - p) <= 70]
                    label_nums = []
                items.append((fname, "%s#%d#%d" % (t, occ, k), tok, set(current), (prev, post, label, sibs, label_nums)))
    return items


def _ctx_score(ctx, line):
    prev, post, label, sibs = ctx[:4]
    low = line.lower()
    words = [w.lower() for w in WORD.findall(prev) if w.lower() not in STOP][-3:]
    words += [w.lower() for w in WORD.findall(re.split(r"\d", post, maxsplit=1)[0]) if w.lower() not in STOP][:2]
    words += [w.lower() for w in WORD.findall(label) if w.lower() not in STOP]
    words = [w.strip("_") for w in words if len(w.strip("_")) >= 2]
    vals = [_num(x) for x in ANY.findall(line)]
    sw = sum(1 for w in set(words) if w in low)
    ss = sum(1 for s in set(sibs) if any(matches(s, v) for v in vals))
    return sw + 2 * ss


def _split(key):
    m, o, k = key.rsplit("#", 2)
    return m, int(o), int(k)


def match(items, lock):
    """Map each doc item to its old lock entry by aligning, per file, the sequence of numbered doc lines in the lock with the
    current one (difflib opcodes): unchanged lines map 1:1 even when a neighbouring repeated row is edited; a replaced block
    pairs lines in order when their masked text is >= 0.6 similar. Returns {item index: entry}."""
    import difflib
    out = {}
    by_file_old, by_file_new = {}, {}
    for e in lock:
        m, o, k = _split(e["key"])
        by_file_old.setdefault(e["file"], {}).setdefault(o, [m, {}])[1][k] = e
    for idx, it in enumerate(items):
        m, o, k = _split(it[1])
        by_file_new.setdefault(it[0], {}).setdefault(o, [m, {}])[1][k] = idx
    for f, new_lines in by_file_new.items():
        old_lines = by_file_old.get(f, {})
        oo, nn = sorted(old_lines), sorted(new_lines)
        sm = difflib.SequenceMatcher(None, [old_lines[o][0] for o in oo], [new_lines[n][0] for n in nn], autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                pairs = list(zip(oo[i1:i2], nn[j1:j2]))
            elif tag == "replace":
                pairs = [(a, b) for a, b in zip(oo[i1:i2], nn[j1:j2])
                         if difflib.SequenceMatcher(None, old_lines[a][0], new_lines[b][0]).ratio() >= 0.6]
            else:
                pairs = []
            for a, b in pairs:
                for k, idx in new_lines[b][1].items():
                    if k in old_lines[a][1]:
                        out[idx] = old_lines[a][1][k]
    return out


def bind(item, outputs, here):
    """Choose the printed position for a doc number: the best context match among lines whose number matches; else a source
    literal. Returns (entry or None, number of equally good candidates on different lines)."""
    fname, key, tok, scripts, ctx = item
    names = sorted(scripts) if scripts else sorted(outputs)
    cands = []
    for n in names:
        for (t, occ, toks, line) in index_output(outputs.get(n, "")):
            if len(ctx) > 4 and ctx[4] and not all(any(_num(x) == float(ln_) for x in toks) for ln_ in ctx[4]):
                continue                                       # the row label (e.g. S=20) must be on the printed line
            for i, x in enumerate(toks):
                if matches(tok, _num(x)):
                    cands.append((_ctx_score(ctx, line), n, t, occ, i))
    if cands:
        best = max(c[0] for c in cands)
        top = [c for c in cands if c[0] == best]
        s, n, t, occ, i = top[0]
        return {"file": fname, "key": key, "tok": tok, "script": n, "kind": "out", "template": t, "occ": occ, "pos": i, "score": s}, \
            len({(c[1], c[2], c[3]) for c in top})
    for n in names:
        src = source_numbers(os.path.join(here, n[:-4] if n.endswith(".txt") else n))
        for i, x in enumerate(src):
            if matches(tok, _num(x)):
                return {"file": fname, "key": key, "tok": tok, "script": n, "kind": "src", "pos": i, "score": 0}, 1
    return None, 0


def read_locked(entry, outputs, here):
    """The value currently at the locked position, or None if the position no longer exists."""
    if entry["kind"] == "src":
        n = entry["script"]
        src = source_numbers(os.path.join(here, n[:-4] if n.endswith(".txt") else n))
        return _num(src[entry["pos"]]) if entry["pos"] < len(src) else None
    for (t, occ, toks, _line) in index_output(outputs.get(entry["script"], "")):
        if t == entry["template"] and occ == entry["occ"]:
            return _num(toks[entry["pos"]]) if entry["pos"] < len(toks) else None
    return None


def label_ok(entry, ctx, outputs):
    """A table row's label numbers (first cell, e.g. S=20) must be printed on the locked line, so swapping row labels fails."""
    nums = ctx[4] if len(ctx) > 4 else []
    if not nums or entry["kind"] != "out":
        return True
    for (t, occ, toks, _line) in index_output(outputs.get(entry["script"], "")):
        if t == entry["template"] and occ == entry["occ"]:
            return all(any(_num(x) == float(n) for x in toks) for n in nums)
    return True


def check(items, lock, outputs, here, allow):
    """Fails: (file, token, reason). A number with no lock entry fails unless it is in ALLOW (quoted, not printed)."""
    mp = match(items, lock)
    fails = []
    for idx, (fname, key, tok, _sc, _ctx) in enumerate(items):
        e = mp.get(idx)
        bare = tok.lstrip("+-").replace(",", "")
        if e is None:
            if bare not in allow:
                fails.append((fname, tok, "not in the lock (new or edited doc number: run --relock and review the diff)"))
            continue
        v = read_locked(e, outputs, here)
        if v is None:
            fails.append((fname, tok, "locked position %s:%s no longer exists (script output changed)" % (e["script"], e.get("template", "src"))))
        elif not matches(tok, v):
            fails.append((fname, tok, "locked value at %s is %s" % (e["script"], v)))
        elif not label_ok(e, _ctx, outputs):
            fails.append((fname, tok, "row label %s is not on the locked printed line of %s" % (_ctx[4], e["script"])))
    return fails


def substitution_rate(items, lock, outputs, here, per_token=8, seed=0):
    """Replace each locked doc number by other values printed or set by the same script (realistic wrong-value edits)."""
    import random
    rnd = random.Random(seed)
    mp = match(items, lock)
    caught = total = 0
    for idx, (fname, key, tok, sc, _ctx) in enumerate(items):
        e = mp.get(idx)
        if e is None:
            continue
        v = read_locked(e, outputs, here)
        n = e["script"]
        pool = {x for x in ANY.findall(outputs.get(n, ""))} | set(source_numbers(os.path.join(here, n[:-4] if n.endswith(".txt") else n)))
        bare = tok.lstrip("+-").replace(",", "")
        dec = len(bare.split(".")[1]) if "." in bare and "e" not in bare else 0
        fmt = lambda x: ("%.*f" % (dec, abs(_num(x)))) if dec else str(int(round(abs(_num(x)))))
        cands = sorted({fmt(x) for x in pool} - {fmt(tok)})
        for c in rnd.sample(cands, min(per_token, len(cands))):
            sub = ("-" if tok.startswith("-") else "") + c
            total += 1
            caught += not (v is not None and matches(sub, v))
    return caught, total


def entry_score(entry, ctx, outputs):
    """Context score of the printed line a lock entry points at (0 for a source literal)."""
    if entry["kind"] != "out":
        return 0
    for (t, occ, _toks, line) in index_output(outputs.get(entry["script"], "")):
        if t == entry["template"] and occ == entry["occ"]:
            return _ctx_score(ctx, line)
    return 0


def _similar(k1, k2):
    import difflib
    return difflib.SequenceMatcher(None, k1, k2).ratio()


def relock(items, outputs, here, old, force=()):
    """Rebuild the lock, guarded. Returns (lock, report); the report lists every decision for review.

    A doc edit must be backed by the script, not the other way round:
      * Same doc position (found by aligning the old and new numbered-line sequences of the file, `match`)
        whose OLD locked position still exists and still prints the OLD value, while the doc token now differs: REFUSED and
        the old entry is kept, so the normal check keeps failing. The script did not change, so the edit contradicts the
        evidence. Override only with --force-relock=<file>:<token> after checking.
      * Old position prints a value matching the new doc token: kept (no silent rebinding).
      * Old position gone or its value changed (the script changed): rebound, but REFUSED if the new binding is a source
        literal where the old one was printed output, or has a lower context score, unless forced.
      * New doc numbers: bound and listed; REFUSED if bound to printed output with no shared context (score 0).
    """
    def forced(fname, tok):
        return ("%s:%s" % (fname, tok)) in force
    mp = match(items, old)
    claimed = set()
    lock = []
    report = {"kept": [], "rebound": [], "added": [], "refused": [], "unbound": [], "removed": [], "ambiguous": []}
    for idx, it in enumerate(items):
        fname, key, tok = it[0], it[1], it[2]
        prev = mp.get(idx)
        if prev is not None:
            claimed.add(id(prev))
            v_old = read_locked(prev, outputs, here)
            if "score" not in prev:
                prev = dict(prev, score=entry_score(prev, it[4], outputs))
            if v_old is not None and matches(tok, v_old) and not label_ok(prev, it[4], outputs) and not forced(fname, tok):
                e2, _n = bind(it, outputs, here)             # same value on a line that DOES carry the row label: rebind
                if e2 is not None and e2["kind"] == "out":
                    lock.append(e2); report["rebound"].append((prev, e2)); continue
                lock.append(dict(prev, key=key))
                report["refused"].append((prev, tok, "row label %s is not printed with this value (labels swapped or edited?)" % (it[4][4],)))
                continue
            if v_old is not None and matches(tok, v_old):
                e = dict(prev, key=key, tok=tok); lock.append(e); report["kept"].append(e); continue
            if v_old is not None and matches(prev["tok"], v_old) and not forced(fname, tok):
                lock.append(dict(prev, key=key))
                report["refused"].append((prev, tok, "the script still prints %s at the locked position; the doc now says %s" % (v_old, tok)))
                continue
        e, n_top = bind(it, outputs, here)
        if e is None:
            report["unbound"].append(it)
            if prev is not None:
                lock.append(dict(prev, key=key))
            continue
        if prev is not None:
            worse = (e["kind"] == "src" and prev["kind"] == "out") or e.get("score", 0) < prev.get("score", 0)
            if worse and not forced(fname, tok):
                lock.append(dict(prev, key=key))
                report["refused"].append((prev, tok, "new binding %s (score %s) is weaker than the old %s (score %s)"
                                          % (e["kind"], e.get("score"), prev["kind"], prev.get("score"))))
                continue
            report["rebound"].append((prev, e))
        else:
            if e["kind"] == "out" and e.get("score", 0) == 0 and not forced(fname, tok):
                report["refused"].append((None, tok, "%s: new number would bind to %s with no shared context" % (fname, e["script"])))
                continue
            report["added"].append(e)
        lock.append(e)
        if n_top > 1:
            report["ambiguous"].append((e, n_top))
    report["removed"] = [e for e in old if id(e) not in claimed]
    return lock, report


def stale(items, lock):
    """Lock entries whose doc number no longer exists (deleted or rewritten doc text)."""
    used = {id(e) for e in match(items, lock).values()}
    return [e for e in lock if id(e) not in used]


def load(path):
    return json.load(open(path, encoding="utf8")) if os.path.exists(path) else []


def save(path, lock):
    json.dump(sorted(lock, key=lambda e: (e["file"], e["key"])), open(path, "w", encoding="utf8"), indent=0, ensure_ascii=False)
