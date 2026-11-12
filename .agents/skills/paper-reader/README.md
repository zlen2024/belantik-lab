# paper-reader

A [Claude Code](https://claude.com/claude-code) skill that reads research
papers and writes notes you can study from and present.

- **Source first.** For arXiv papers it downloads the PDF and the LaTeX source,
  and reads equations, tables and figures from the source, where they are
  exact.
- **Three depths.** `scan` to decide whether a paper is worth reading,
  `understand` (default) to explain the main thread and its evidence,
  `reconstruct` to rebuild derivations and find gaps.
- **Bilingual notes.** Every note is saved in Chinese and in English. The
  Chinese version keeps technical terms in English.
- **Notes or slides.** Markdown by default; HTML on request, with a slide mode
  (press S) that shows only key points and figures.
- **One folder per paper.** `paper-notes/<paper>/` holds `paper.pdf`,
  `source/`, the notes and the figures they use.

## Install

```bash
git clone --recurse-submodules https://github.com/Alwin-Yang/paper-reader ~/.claude/skills/paper-reader
```

The skill uses `bash`, `curl`, `tar` and `python3`. To convert PDF figures to
PNG it uses poppler (`pdftoppm`) or Ghostscript when installed, and falls back
to macOS `sips`, which renders some figures blank or cropped. Installing one of
them is recommended:

```bash
brew install poppler      # or: brew install ghostscript
```

## Use

Ask Claude Code in any project, for example:

```text
精读 https://arxiv.org/abs/2410.24164，输出 HTML，我要组会讲
Quickly screen https://arxiv.org/abs/2406.09246 — is it worth a close read?
Diffusion Policy (arXiv 2303.04137) 的 receding horizon 部分帮我深入重建一下
```

Result:

```
paper-notes/pi0/
  paper.pdf
  source/          LaTeX source
  index.zh.html    Chinese notes; press S for slides
  index.en.html    English notes
  figures/
```

If the project is a git repository, add `paper-notes/` to `.gitignore`; the
skill reminds you when it is missing.

## Layout

```
SKILL.md                     workflow
references/reading-method.md depths, note structure, paper types
references/markdown-output.md
references/html-output.md
assets/note-template.html    HTML template with slide mode
scripts/fetch_arxiv_source.sh  PDF + LaTeX source + figures
scripts/build_note.py        wrap a note body in the template
scripts/check_note.py        check a paper folder
scripts/upstream_diff.sh     review upstream changes
upstream/                    original skills as git submodules
```

## Credits

Built from two MIT-licensed skills, kept as submodules under `upstream/`:

- [paper-reading-skill](https://github.com/kingslayer-bot/paper-reading-skill)
  by kingslayer-bot: three reading depths and the four-question note
  structure, based on S. Keshav's "How to Read a Paper".
- [read-arxiv-paper](https://github.com/karpathy/nanochat/tree/master/.claude/skills/read-arxiv-paper)
  in nanochat by Andrej Karpathy: read the arXiv LaTeX source, not the PDF.

See [UPSTREAM.md](UPSTREAM.md) for what came from where and how to pull
upstream changes.

## License

MIT. See [LICENSE](LICENSE).
