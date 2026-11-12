#!/usr/bin/env python3
"""Check a paper folder for the mechanical rules of the paper-reader skill.

Usage: python3 check_note.py paper-notes/<paper-short-name>

Checks: paper.pdf and source/ are present; notes exist as a zh/en pair; the
two versions match in structure (sections, headings, tables, display math,
images); local image links resolve; Markdown metadata is complete; the zh
version keeps technical terms in English. It does not judge whether the
reading is correct or deep enough.

Exit 0 when there are no errors, 1 otherwise. Warnings do not fail.
"""

import re
import sys
from pathlib import Path

# Chinese renderings of technical terms that the zh notes should leave in
# English. Kept short and unambiguous to avoid false alarms.
ZH_TERMS = {
    "预训练": "pre-training", "后训练": "post-training", "微调": "fine-tuning",
    "消融": "ablation", "基线": "baseline", "注意力掩码": "attention mask",
    "自注意力": "self-attention", "交叉注意力": "cross-attention",
    "动作块": "action chunk", "动作分块": "action chunking", "动作专家": "action expert",
    "流匹配": "flow matching", "扩散模型": "diffusion model", "自回归": "autoregressive",
    "离散化": "discretization", "损失函数": "loss function", "反向传播": "backpropagation",
    "过拟合": "overfitting", "编码器": "encoder", "解码器": "decoder",
    "嵌入向量": "embedding", "世界模型": "world model", "视觉语言模型": "vision-language model",
    "强化学习": "reinforcement learning", "模仿学习": "imitation learning",
    "行为克隆": "behavior cloning", "跨本体": "cross-embodiment", "本体感知": "proprioception",
    "开环": "open-loop", "闭环": "closed-loop", "时间集成": "temporal ensembling",
}
MD_FIELDS = ("title", "paper_type", "reading_depth", "source_language", "tags", "language")
DEPTHS = ("scan", "understand", "reconstruct")


class Report:
    def __init__(self):
        self.errors, self.warnings = [], []

    def error(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


def strip_code(text):
    return re.sub(r"(?ms)^(`{3,}|~{3,})[^\n]*\n.*?^\1\s*$", "", text)


def zh_term_check(text, name, rep):
    visible = re.sub(r"<[^>]+>", " ", strip_code(text))
    found = {zh: visible.count(zh) for zh in ZH_TERMS if zh in visible}
    for zh, n in sorted(found.items(), key=lambda kv: -kv[1]):
        rep.error("%s: Chinese term '%s' x%d; write '%s'" % (name, zh, n, ZH_TERMS[zh]))


def check_images(text, folder, name, rep):
    targets = re.findall(r'<img[^>]+src="([^"]+)"', text)
    targets += re.findall(r"!\[[^\]]*\]\(([^)\s]+)", text)
    for t in targets:
        if re.match(r"^[a-z]+:", t):
            continue
        if not (folder / t).exists():
            rep.error("%s: image not found: %s" % (name, t))
    return sorted(targets)


def check_html_pair(folder, rep):
    zh, en = folder / "index.zh.html", folder / "index.en.html"
    texts = {}
    for path, lang, other in ((zh, "zh-CN", "index.en.html"), (en, "en", "index.zh.html")):
        t = path.read_text(encoding="utf-8")
        texts[path.name] = t
        if '<html lang="%s">' % lang not in t:
            rep.error("%s: <html lang> should be %s" % (path.name, lang))
        if 'class="lang-switch" href="%s"' % other not in t:
            rep.error("%s: missing language switch link to %s" % (path.name, other))
        left = re.findall(r"\b(?:PAPER_[A-Z_]+|OTHER_LANG_[A-Z]+|ONE_PARAGRAPH_OVERVIEW|SECTION_TITLE|KEY_POINT)\b", t)
        if left:
            rep.error("%s: unfilled template placeholders: %s" % (path.name, ", ".join(sorted(set(left)))))
        if "renderMathInElement" not in t:
            rep.error("%s: not built from the note template (no KaTeX)" % path.name)
        for sec in re.findall(r'<section class="slide"[^>]*>.*?</section>', t, re.S):
            if 'class="takeaway"' not in sec or 'class="detail"' not in sec:
                sid = re.search(r'id="([^"]+)"', sec)
                rep.warn("%s: section %s lacks takeaway or detail" % (path.name, sid.group(1) if sid else "?"))
    a, b = texts["index.zh.html"], texts["index.en.html"]
    ids = [re.findall(r'<section class="slide" id="([^"]+)"', x) for x in (a, b)]
    if not ids[0]:
        rep.error("index.zh.html: no <section class=\"slide\" id=...> blocks")
    if ids[0] != ids[1]:
        rep.error("section ids differ between zh and en: %s vs %s" % (ids[0], ids[1]))
    for label, pat in (("tables", r"<table"), ("display math blocks", r"\$\$"), ("figures", r"<img ")):
        n = [len(re.findall(pat, x)) for x in (a, b)]
        if n[0] != n[1]:
            rep.error("%s differ between zh (%d) and en (%d)" % (label, n[0], n[1]))
    imgs = [check_images(x, folder, name, rep) for name, x in texts.items()]
    if imgs[0] != imgs[1]:
        rep.error("zh and en reference different images")
    zh_term_check(a, "index.zh.html", rep)


def front_matter(text, name, rep):
    m = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)", text, re.S)
    if not m:
        rep.error("%s: missing YAML front matter" % name)
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        kv = re.match(r"^([A-Za-z_]+):\s*(.*)$", line)
        if kv:
            meta[kv.group(1)] = kv.group(2).strip()
        elif re.match(r"^\s+-\s+\S", line) and meta:
            last = list(meta)[-1]
            meta[last] = (meta[last] + " [list]").strip()
    return meta, text[m.end():]


def check_md_pair(folder, rep):
    bodies, depths = {}, {}
    for name, lang, other in (("note.zh.md", "zh", "note.en.md"), ("note.en.md", "en", "note.zh.md")):
        text = (folder / name).read_text(encoding="utf-8")
        meta, body = front_matter(text, name, rep)
        for field in MD_FIELDS:
            if not meta.get(field):
                rep.error("%s: missing metadata field '%s'" % (name, field))
        if meta.get("language") and meta["language"].strip("'\"") != lang:
            rep.error("%s: language should be %s" % (name, lang))
        depth = meta.get("reading_depth", "").strip("'\"")
        if depth and depth not in DEPTHS:
            rep.error("%s: reading_depth must be one of %s" % (name, ", ".join(DEPTHS)))
        depths[name] = depth
        if other not in body:
            rep.error("%s: no link to %s" % (name, other))
        visible = strip_code(body)
        heads = re.findall(r"(?m)^(#{1,6})\s+\S", visible)
        bodies[name] = (visible, heads)
        hs = list(re.finditer(r"(?m)^(#{1,6})\s+(.+)$", visible))
        for i, h in enumerate(hs):
            end = hs[i + 1].start() if i + 1 < len(hs) else len(visible)
            nxt_deeper = i + 1 < len(hs) and len(hs[i + 1].group(1)) > len(h.group(1))
            if not visible[h.end():end].strip() and not nxt_deeper:
                rep.error("%s: empty section '%s'" % (name, h.group(2).strip()))
    (zv, zh_heads), (ev, en_heads) = bodies["note.zh.md"], bodies["note.en.md"]
    if [len(h) for h in zh_heads] != [len(h) for h in en_heads]:
        rep.error("heading structure differs between zh (%d headings) and en (%d)" % (len(zh_heads), len(en_heads)))
    if depths["note.zh.md"] != depths["note.en.md"]:
        rep.error("reading_depth differs between zh and en")
    n = [len(re.findall(r"\$\$", x)) for x in (zv, ev)]
    if n[0] != n[1]:
        rep.error("display math blocks differ between zh (%d) and en (%d)" % tuple(n))
    imgs = [check_images(x, folder, name, rep) for name, x in (("note.zh.md", zv), ("note.en.md", ev))]
    if imgs[0] != imgs[1]:
        rep.error("zh and en reference different images")
    zh_term_check(zv, "note.zh.md", rep)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    folder = Path(sys.argv[1])
    rep = Report()
    if not folder.is_dir():
        print("Not a folder: %s" % folder)
        return 2

    pdf = folder / "paper.pdf"
    if not pdf.exists():
        rep.error("paper.pdf is missing")
    elif not pdf.read_bytes()[:5].startswith(b"%PDF"):
        rep.error("paper.pdf is not a PDF")
    src = folder / "source"
    if not src.is_dir():
        rep.warn("source/ is missing (fine only if the paper has no public LaTeX source)")
    elif not list(src.rglob("*.tex")):
        rep.warn("source/ has no .tex files")

    html = [(folder / n).exists() for n in ("index.zh.html", "index.en.html")]
    md = [(folder / n).exists() for n in ("note.zh.md", "note.en.md")]
    if not any(html + md):
        rep.error("no notes found (index.zh.html/index.en.html or note.zh.md/note.en.md)")
    if any(html) and not all(html):
        rep.error("HTML notes must come as a pair: index.zh.html and index.en.html")
    if any(md) and not all(md):
        rep.error("Markdown notes must come as a pair: note.zh.md and note.en.md")
    if all(html):
        check_html_pair(folder, rep)
    if all(md):
        check_md_pair(folder, rep)

    for w in rep.warnings:
        print("WARN: " + w)
    for e in rep.errors:
        print("ERROR: " + e)
    print("%s: %d errors, %d warnings" % ("FAIL" if rep.errors else "PASS", len(rep.errors), len(rep.warnings)))
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
