"""Copy frozen top-level files to local archive without editing originals."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    args = parser.parse_args()
    target = ROOT / 'archives' / 'legacy-root-files'
    if target.exists():
        raise FileExistsError('Destination already exists; inspect before retry')
    entries = []
    for line in args.inventory.open(encoding='utf-8'):
        item = json.loads(line)
        if len(Path(item['path']).parts) == 1 and item['type'] == 'file':
            entries.append(item)
    if not entries or any(x.get('changed_during_hash') for x in entries):
        raise ValueError('No frozen files or source changed during inventory')
    target.mkdir(parents=True)
    report = {'status': 'incomplete', 'files_verified': 0, 'error': None,
              'inventory': str(args.inventory.resolve()), 'off_device_backup_verified': False}
    try:
        for item in entries:
            source, dest = ROOT / item['path'], target / item['path']
            before = source.stat(follow_symlinks=False)
            if (before.st_size, before.st_mtime_ns) != (item['size'], item['mtime_ns']):
                raise RuntimeError(f'Source changed since inventory: {item["path"]}')
            shutil.copy2(source, dest)
            after = source.stat(follow_symlinks=False)
            if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns) or digest(dest) != item['sha256']:
                raise RuntimeError(f'Copy mismatch: {item["path"]}')
            report['files_verified'] += 1
        report['status'] = 'contents_sha256_verified'
    except Exception as exc:
        report['error'] = str(exc)
    finally:
        (ROOT / 'archives' / 'reports').mkdir(exist_ok=True)
        (ROOT / 'archives' / 'reports' / 'legacy-root-files.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'contents_sha256_verified' else 2


if __name__ == '__main__':
    raise SystemExit(main())
