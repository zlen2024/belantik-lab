#!/usr/bin/env python3
"""Validate a single Malay Markdown note using the upstream structural helpers.

This checks structure and local images, not scientific grounding or Malay fluency.
"""
import argparse
import re
from pathlib import Path
from check_note import DEPTHS, MD_FIELDS, Report, check_images, front_matter, strip_code


def validate(folder):
    folder = Path(folder)
    rep = Report()
    path = folder / 'note.ms.md'
    if not path.is_file():
        rep.error('note.ms.md is missing')
        return rep
    pdf = folder / 'paper.pdf'
    if not pdf.is_file() or not pdf.read_bytes().startswith(b'%PDF'):
        rep.error('paper.pdf is missing or is not a PDF')
    text = path.read_text(encoding='utf-8')
    meta, body = front_matter(text, path.name, rep)
    for field in MD_FIELDS:
        if not meta.get(field):
            rep.error('missing metadata: ' + field)
    if meta.get('language', '').strip('\"\'') != 'ms':
        rep.error('language must be ms')
    if meta.get('reading_depth', '').strip('\"\'') not in DEPTHS:
        rep.error('reading_depth must be scan, understand or reconstruct')
    visible = strip_code(body)
    headings = list(re.finditer(r'(?m)^(#{1,6})\s+(.+)$', visible))
    if not headings:
        rep.error('note has no headings')
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(visible)
        child = i + 1 < len(headings) and len(headings[i + 1].group(1)) > len(heading.group(1))
        if not visible[heading.end():end].strip() and not child:
            rep.error('empty section: ' + heading.group(2))
    check_images(body, folder, path.name, rep)
    if visible.count('$$') % 2:
        rep.error('unpaired display math delimiter')
    return rep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    report = validate(args.folder)
    for error in report.errors:
        print('ERROR: ' + error)
    print(('FAIL' if report.errors else 'PASS') + ': structural checks only; inspect evidence and language separately.')
    return bool(report.errors)


if __name__ == '__main__':
    raise SystemExit(main())
