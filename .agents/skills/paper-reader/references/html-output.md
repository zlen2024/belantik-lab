# HTML output with slide mode

Use this format when the user asks for HTML, a web page, slides, or material
for a talk or group meeting. One file works as a detailed article and, after
pressing S, as a slide deck: each section shows only its title, key points and
figure, while the full explanation and speaker notes stay hidden.

## Files

```
paper-notes/<paper-short-name>/
  paper.pdf
  source/
  index.zh.html
  index.en.html
  figures/       only figures the notes use, shared by both versions
```

Write only the content inside `<main>` for each language, then wrap it with
the template:

```bash
python3 ~/.claude/skills/paper-reader/scripts/build_note.py body.zh.html paper-notes/<name>/index.zh.html --lang zh --title "π0 论文笔记"
python3 ~/.claude/skills/paper-reader/scripts/build_note.py body.en.html paper-notes/<name>/index.en.html --lang en --title "π0 Paper Notes"
```

Keep the body files in a scratch location, not in the paper folder. The
template (`assets/note-template.html`) supplies the styles, light and dark
themes, KaTeX, Mermaid and the slide-mode script; do not copy or change them
unless the user asks. The script refuses to write a file that still has
template placeholders.

Copy the figures you use from the fetch cache (`~/.cache/arxiv-papers/<id>/figures/`)
into the folder's `figures/` and reference them as `figures/<name>.png`.

## Body structure

Start the body with the header, then one `<section class="slide">` per idea:

```html
<header class="paper">
  <a class="lang-switch" href="index.en.html">English</a>   <!-- in en: href="index.zh.html">中文 -->
  <h1>Full paper title</h1>
  <div class="meta">
    Authors, year · <a href="https://arxiv.org/abs/ID">arXiv:ID</a><br>
    paper_type: … · reading_depth: … · source_language: … · tags: …
  </div>
  <p>One-paragraph overview.</p>
  <div class="hint">按 S 切换幻灯片模式，方向键翻页，N 显示讲解要点。</div>
  <!-- en: Press S for slide mode, arrow keys to move, N for speaker notes. -->
</header>

<section class="slide" id="problem">
  <h2>Section title</h2>
  <div class="takeaway"><ul><li>Key point</li></ul></div>
  <figure>
    <img src="figures/overview.png" alt="What the figure shows">
    <figcaption>Fig. 3: caption</figcaption>
  </figure>
  <div class="detail">
    <p>Full explanation.</p>
    <p class="src">Source: Sec. IV, Eq. (2)</p>
  </div>
  <aside class="notes">What to say when presenting.</aside>
</section>
```

Each section covers one idea that can stand as one slide; do not copy the four
question parts as four huge sections. The four-question structure still sets
the order and coverage. A typical order for a method paper:

1. The problem, and what earlier approaches got wrong
2. The core idea, in one sentence
3. Architecture, with the paper's figure or a Mermaid diagram
4. Key equations or mechanisms (several sections if needed)
5. Data and training recipe
6. Key results, each paired with its evidence
7. Strength of evidence, limitations, takeaways

Inside each section:

- `.takeaway` is what appears on the slide: 3 to 5 short points that each fit
  on one line, plus one figure when there is one.
- `.detail` is the full explanation, hidden in slide mode. End it with
  `<p class="src">` giving the section, equation, figure or table it comes
  from.
- `.notes` is what to say when presenting: transitions and questions the
  audience is likely to ask. Shown in slide mode with N.

Use the same section `id` values in both languages; the checker compares them.

## Formatting

- Write math as `$...$` and `$$...$$`; KaTeX renders it. Inside math, write
  `<` and `>` as `\lt` and `\gt`, and `&` as `&amp;`, so the browser does not
  read them as HTML.
- Diagrams use `<pre class="mermaid">…</pre>`; no LaTeX in node labels.
- Exact tensor or token layouts use a plain `<pre>` block with ASCII and
  dimensions.
- Tables only for small sets of comparable numbers; keep conditions and units.
- The header's meta line replaces the Markdown metadata.

## In the chat

Do not paste the page. Give links to both files, mention the S key, summarize
the problem, method and main result in a few sentences, and say what could not
be read or verified.

If the user also wants Markdown, save `note.zh.md` and `note.en.md` with the
same content as the `.detail` parts.
