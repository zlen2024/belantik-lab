# Markdown output

The default format. Notes are standard Markdown, readable in any Markdown
viewer and in Obsidian.

## Files

```
paper-notes/<paper-short-name>/
  paper.pdf
  source/
  note.zh.md
  note.en.md
  figures/       only figures the notes embed, shared by both versions
```

Update the same files when the same paper is read again.

## Metadata

Each file starts with YAML front matter:

```yaml
---
title: "π0: A Vision-Language-Action Flow Model for General Robot Control"
paper_type: method
reading_depth: understand        # scan | understand | reconstruct
source_language: en
language: zh                     # zh in note.zh.md, en in note.en.md
tags:
  - VLA
  - flow matching
arxiv: "2410.24164"              # optional, only when verified
authors: "Black, Brown, ..."     # optional, only when verified
---
```

Required: `title`, `paper_type`, `reading_depth`, `source_language`, `tags`
(a list), `language`. Write authors, year, venue, DOI and URL only when
verified. Use the real current date if you record one.

Right below the front matter, link to the other version, for example
`[English](note.en.md)` in the Chinese file and `[中文](note.zh.md)` in the
English one.

## Body

- Standard Markdown with `$...$` and `$$...$$` math.
- Use `[[wikilinks]]` only when the user asks for Obsidian links, and write to
  an Obsidian vault only when the user gives and authorizes the path.
- Figures may be described by number instead of embedded. When embedded, copy
  the PNG from the fetch cache into `figures/` and use a relative path. Never
  link a file that does not exist.

## In the chat

Show the full Chinese note in the chat (without the YAML block), not a summary,
and give links to both files. If the note is too long for one message, show it
in consecutive parts by section. If files cannot be written, say so instead of
claiming they exist.
