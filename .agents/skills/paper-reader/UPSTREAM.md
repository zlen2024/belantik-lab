# Upstream sources

This skill merges two MIT-licensed skills. Both are kept as git submodules
under `upstream/`, pinned to the commit the current files were derived from,
so upstream changes can be reviewed and ported.

| Submodule | Upstream | Pinned commit | Used for |
|---|---|---|---|
| `upstream/paper-reading-skill` | [kingslayer-bot/paper-reading-skill](https://github.com/kingslayer-bot/paper-reading-skill) | `aec9608` | `references/reading-method.md` (three depths, four-question structure, emphasis by paper type, targeted reading); evidence rules in `SKILL.md`; the idea behind `scripts/check_note.py` |
| `upstream/nanochat` | [karpathy/nanochat](https://github.com/karpathy/nanochat), file `.claude/skills/read-arxiv-paper/SKILL.md` | `92d63d4` | Step 1 and step 2 of `SKILL.md` (fetch the arXiv `/src/` archive, find the entry point, read the source rather than the PDF); `scripts/fetch_arxiv_source.sh` |

`nanochat` is a whole repository, so it is a shallow submodule; only one file
in it matters here.

## What is new in this skill

Not from either upstream:

- One folder per paper holding `paper.pdf`, the LaTeX `source/` and the notes.
- Notes always written in Chinese and English, with technical terms kept in
  English in the Chinese version.
- HTML output with a slide mode (`assets/note-template.html`,
  `scripts/build_note.py`, `references/html-output.md`).
- Figure conversion to PNG and PDF download in the fetch script.
- `scripts/check_note.py` checks for the folder and the zh/en pair.

## Updating

```bash
git submodule update --init          # first time after cloning
bash scripts/upstream_diff.sh        # what changed upstream, relevant files only
```

Port what is useful into this skill's files, then move the pins and record the
new commits in the table above:

```bash
git submodule update --remote
git add upstream UPSTREAM.md SKILL.md references scripts
git commit -m "Sync upstream changes"
```
