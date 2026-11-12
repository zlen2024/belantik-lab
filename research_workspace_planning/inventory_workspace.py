"""Freeze a read-only SHA-256 inventory without traversing Windows reparse points."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'research_workspace_planning' / 'inventory'
EXCLUDE_TOP = {'_safety_backup_20261112_word_energy', 'word_energy_safety_backup_20261112.zip',
               'projects', 'archives', '.git'}
EXCLUDE_REL = {'research_workspace_planning/inventory'}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    manifest = OUTPUT / f'source_{stamp}.jsonl'
    summary = {'started_utc': stamp, 'root': str(ROOT), 'manifest': str(manifest),
               'excluded_generated': sorted(EXCLUDE_TOP | EXCLUDE_REL),
               'files': 0, 'directories': 0, 'reparse_points': 0,
               'bytes': 0, 'errors': 0, 'changed_during_hash': 0}

    with manifest.open('x', encoding='utf-8') as f:
        def write(record):
            f.write(json.dumps(record, ensure_ascii=True) + '\n')

        def scan(path):
            rel = path.relative_to(ROOT).as_posix()
            if (len(path.relative_to(ROOT).parts) == 1 and path.name in EXCLUDE_TOP) or rel in EXCLUDE_REL:
                write({'path': rel, 'type': 'generated_exclusion'})
                return
            try:
                info = path.stat(follow_symlinks=False)
                attrs = getattr(info, 'st_file_attributes', 0)
                record = {'path': rel, 'size': info.st_size, 'mtime_ns': info.st_mtime_ns,
                          'attributes': attrs}
                if attrs & 0x400 or path.is_symlink():
                    record.update(type='reparse_point', tag=getattr(info, 'st_reparse_tag', None))
                    try:
                        record['target'] = os.readlink(path)
                    except OSError as exc:
                        record['target_error'] = str(exc)
                        summary['errors'] += 1
                    summary['reparse_points'] += 1
                    write(record)
                elif path.is_dir():
                    summary['directories'] += 1
                    record['type'] = 'directory'
                    write(record)
                    with os.scandir(path) as entries:
                        names = sorted((entry.name for entry in entries), key=str.casefold)
                    for name in names:
                        scan(path / name)
                elif path.is_file():
                    digest = hashlib.sha256()
                    with path.open('rb') as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            digest.update(block)
                    after = path.stat(follow_symlinks=False)
                    changed = (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                    record.update(type='file', sha256=digest.hexdigest(), changed_during_hash=changed)
                    summary['files'] += 1
                    summary['bytes'] += info.st_size
                    summary['changed_during_hash'] += int(changed)
                    write(record)
                else:
                    summary['errors'] += 1
                    record['type'] = 'unsupported'
                    write(record)
            except (OSError, ValueError) as exc:
                summary['errors'] += 1
                write({'path': rel, 'type': 'error', 'error': str(exc)})

        for path in sorted(ROOT.iterdir(), key=lambda p: p.name.casefold()):
            scan(path)
    summary['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    summary['complete_for_frozen_scope'] = not (summary['errors'] or summary['changed_during_hash'])
    out = OUTPUT / f'summary_{stamp}.json'
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0 if summary['complete_for_frozen_scope'] else 2


if __name__ == '__main__':
    sys.exit(main())
