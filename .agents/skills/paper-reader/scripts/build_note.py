#!/usr/bin/env python3
"""Wrap a note body in the HTML template.

Usage:
  python3 build_note.py BODY.html OUT.html --lang zh|en --title "Short title"

BODY.html holds everything that goes inside <main>: the <header class="paper">
and the <section class="slide"> blocks. The template supplies the styles,
KaTeX, Mermaid and the slide-mode script, so notes never copy them by hand.
"""

import argparse
import re
import sys
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent / "assets" / "note-template.html"
# Placeholders that must not survive into a finished note.
PLACEHOLDER = re.compile(
    r"\b(?:PAPER_[A-Z_]+|OTHER_LANG_[A-Z]+|HINT|LANG|ARXIV_ID|AUTHORS|YEAR|TYPE|DEPTH|TAGS|"
    r"ONE_PARAGRAPH_OVERVIEW|SECTION_TITLE|KEY_POINT|FIGURE_DESCRIPTION|FULL_EXPLANATION|"
    r"WHAT_TO_SAY_WHEN_PRESENTING|CAPTION)\b"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("body")
    parser.add_argument("out")
    parser.add_argument("--lang", choices=["zh", "en"], required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    template = TEMPLATE.read_text(encoding="utf-8")
    body = Path(args.body).read_text(encoding="utf-8")
    html = re.sub(r"<main>.*</main>", lambda m: "<main>\n" + body.strip() + "\n</main>", template, flags=re.S)
    html = html.replace('<html lang="LANG">', '<html lang="%s">' % ("zh-CN" if args.lang == "zh" else "en"))
    html = html.replace("<title>PAPER_SHORT_TITLE</title>", "<title>%s</title>" % args.title)

    leftover = sorted(set(PLACEHOLDER.findall(html)))
    if leftover:
        print("Unfilled placeholders: " + ", ".join(leftover), file=sys.stderr)
        return 1
    Path(args.out).write_text(html, encoding="utf-8")
    print("Wrote %s (%d bytes)" % (args.out, len(html)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
