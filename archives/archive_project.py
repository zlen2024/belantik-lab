"""Copy a frozen-inventory project, verify hashes; never delete source or partial data."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'archives' / 'legacy-projects'
REPORTS = ROOT / 'archives' / 'reports'


def fs_path(path):
    """Extended-length local Windows path; callers validate workspace boundaries first."""
    raw = os.path.abspath(str(path))
    if os.name != 'nt' or raw.startswith('\\\\?\\'):
        return raw
    if raw.startswith('\\\\'):
        return '\\\\?\\UNC\\' + raw[2:]
    return '\\\\?\\' + raw


def digest(path):
    result = hashlib.sha256()
    with open(fs_path(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='single top-level legacy directory name')
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='verify existing partial files and resume without overwriting')
    args = parser.parse_args()
    name = args.name
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('Expected one top-level directory name')
    source = ROOT / name
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    target = ARCHIVE / name
    if source.is_symlink() or source.is_junction() or not source.is_dir():
        raise ValueError('Source must be an ordinary top-level directory')
    if not source.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Source escaped workspace')
    if not target.resolve().is_relative_to(ARCHIVE.resolve()):
        raise ValueError('Target escaped intended archive directory')
    resuming = target.exists() and args.resume
    if resuming:
        initial_report = REPORTS / f'{name}.json'
        if not initial_report.is_file():
            raise FileNotFoundError('Partial archive has no initial report')
        previous = json.loads(initial_report.read_text(encoding='utf-8'))
        if previous.get('status') != 'incomplete' or previous.get('inventory') != str(args.inventory.resolve()):
            raise ValueError('Resume requires an incomplete report from the same frozen inventory')
    elif target.exists():
        # A different batch may be copying this exact project. Never overwrite partial output.
        report_path = REPORTS / f'{name}.json'
        for _ in range(180):
            if report_path.exists():
                previous = json.loads(report_path.read_text(encoding='utf-8'))
                if (previous.get('status') == 'contents_sha256_verified'
                        and previous.get('inventory') == str(args.inventory.resolve())):
                    print(name, 'already verified from the same inventory; skipping', flush=True)
                    return 0
                raise FileExistsError('Existing archive has not been verified; inspect report')
            time.sleep(10)
        raise TimeoutError('Existing archive has no completed report; inspect partial copy')

    entries = []
    with args.inventory.open(encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            if record['path'] == name or record['path'].startswith(name + '/'):
                entries.append(record)
    if not entries:
        raise ValueError('Source absent from frozen inventory')
    invalid = [r for r in entries if r['type'] not in ('file', 'directory') or r.get('changed_during_hash')]
    if invalid:
        raise ValueError(f'{len(invalid)} reparse/error/changed records need explicit resolution')
    expected = {r['path'] for r in entries}

    def check_tree():
        actual = set()
        for current, dirs, files in os.walk(source, followlinks=False, onerror=lambda e: (_ for _ in ()).throw(e)):
            folder = Path(current)
            actual.add(folder.relative_to(ROOT).as_posix())
            for part in dirs + files:
                path = folder / part
                info = path.stat(follow_symlinks=False)
                if getattr(info, 'st_file_attributes', 0) & 0x400 or path.is_symlink():
                    raise RuntimeError(f'New reparse point requires review: {path.relative_to(ROOT)}')
                actual.add(path.relative_to(ROOT).as_posix())
        if actual != expected:
            raise RuntimeError(f'Source paths changed: {len(actual-expected)} added, {len(expected-actual)} missing')

    check_tree()
    bytes_needed = sum(r['size'] for r in entries if r['type'] == 'file')
    if shutil.disk_usage(ARCHIVE).free < bytes_needed * 1.05 + 256 * 1024 * 1024:
        raise RuntimeError('Insufficient destination space for copy and safety margin')

    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report = {'source': str(source), 'destination': str(target), 'inventory': str(args.inventory.resolve()),
              'started_utc': stamp, 'status': 'incomplete', 'files_verified': 0, 'bytes': 0,
              'error': None, 'off_device_backup_verified': False}
    report_path = (REPORTS / f'{name}.retry-{datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}.json'
                   if resuming else REPORTS / f'{name}.json')
    if report_path.exists():
        raise FileExistsError('Report exists; do not overwrite historical status')
    try:
        if not resuming:
            target.mkdir()
        for item in entries:
            rel = Path(item['path']).relative_to(name)
            original = source / rel
            copy = target / rel
            if '..' in rel.parts or not copy.absolute().is_relative_to(target.absolute()):
                raise ValueError('Copy target escaped archive')
            if item['type'] == 'directory':
                os.makedirs(fs_path(copy), exist_ok=True)
                continue
            before = os.stat(fs_path(original), follow_symlinks=False)
            if (before.st_size, before.st_mtime_ns) != (item['size'], item['mtime_ns']):
                raise RuntimeError(f'Source changed after inventory: {item["path"]}')
            os.makedirs(fs_path(copy.parent), exist_ok=True)
            if os.path.exists(fs_path(copy)):
                if not resuming or digest(copy) != item['sha256']:
                    raise RuntimeError(f'Existing destination differs; no overwrite: {item["path"]}')
            else:
                shutil.copy2(fs_path(original), fs_path(copy))
            after = os.stat(fs_path(original), follow_symlinks=False)
            if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                raise RuntimeError(f'Source changed during copying: {item["path"]}')
            if digest(copy) != item['sha256']:
                raise RuntimeError(f'Copy hash mismatch: {item["path"]}')
            report['files_verified'] += 1
            report['bytes'] += item['size']
            if report['files_verified'] % 1000 == 0:
                print(name, 'files verified:', report['files_verified'], flush=True)
        check_tree()
        expected_dest = {Path(r['path']).relative_to(name).as_posix() for r in entries}
        actual_dest = set()
        target_fs = Path(fs_path(target))
        for current, dirs, files in os.walk(target_fs, followlinks=False, onerror=lambda e: (_ for _ in ()).throw(e)):
            parent = Path(current)
            actual_dest.add(parent.relative_to(target_fs).as_posix())
            for part in dirs + files:
                path = parent / part
                info = path.stat(follow_symlinks=False)
                if getattr(info, 'st_file_attributes', 0) & 0x400 or path.is_symlink():
                    raise RuntimeError('Unexpected reparse point in destination')
                actual_dest.add(path.relative_to(target_fs).as_posix())
        if actual_dest != expected_dest:
            raise RuntimeError(f'Destination paths mismatch: {len(actual_dest-expected_dest)} extra, {len(expected_dest-actual_dest)} missing')
        report['status'] = 'contents_sha256_verified'
    except Exception as exc:
        report['error'] = str(exc)
    finally:
        report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'contents_sha256_verified' else 2


if __name__ == '__main__':
    sys.exit(main())
