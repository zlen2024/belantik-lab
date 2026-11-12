"""Create and verify a ZIP64 backup without following filesystem reparse points.

Legacy originals are omitted only because their locally verified archive copies
are included; the embedded manifest records that mapping and all link targets.
The ZIP is not encrypted and is not an off-device backup until copied elsewhere.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PARTIAL = ROOT / 'research_workspace_full_backup_20261112.zip.partial'
FINAL = ROOT / 'research_workspace_full_backup_20261112.zip'
REPORT = ROOT / 'archives' / 'reports' / 'full_backup_20261112.json'
ARCHIVE_INDEX = ROOT / 'archives' / 'index.yaml'
STORE_EXTENSIONS = {'.zip', '.gz', '.7z', '.rar', '.mp4', '.mp3', '.jpg', '.jpeg', '.png',
                    '.webm', '.pdf', '.woff2', '.pack', '.npz'}


def fs_path(path):
    value = os.path.abspath(str(path))
    if os.name != 'nt' or value.startswith('\\\\?\\'):
        return value
    return '\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value


def sha_file(path):
    h = hashlib.sha256()
    with open(fs_path(path), 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if FINAL.exists() or PARTIAL.exists() or REPORT.exists():
        raise FileExistsError('Backup output or report already exists; refusing overwrite')
    index = json.loads(ARCHIVE_INDEX.read_text(encoding='utf-8'))
    if any(item['status'] != 'contents_sha256_verified' for item in index['entries']):
        raise RuntimeError('Not every local archive entry has passed content verification')
    mappings = {item['original_path']: 'archives/' + item['archive_path']
                for item in index['entries'] if item['original_path'] != '.'}
    for original, archived in mappings.items():
        if not (ROOT / original).is_dir() or not (ROOT / archived).is_dir():
            raise RuntimeError('A legacy source or verified archive directory is missing')
    free = shutil.disk_usage(ROOT).free
    # Conservative upper bound from original archive inventory plus 10 GiB for
    # the active workspaces, skills, metadata, ZIP overhead and margin.
    if free < 32 * 1024**3:
        raise RuntimeError('Less than 32 GiB free; no backup written')

    # A legacy original may only be omitted when it still matches the frozen
    # inventory used to verify its archive copy. Reject new/missing paths too.
    inventory_dir = ROOT / 'research_workspace_planning' / 'inventory'
    summaries = sorted(inventory_dir.glob('summary_*.json'))
    if not summaries:
        raise RuntimeError('Frozen source inventory summary missing')
    inventory = json.loads(summaries[-1].read_text(encoding='utf-8'))
    if not inventory.get('complete_for_frozen_scope'):
        raise RuntimeError('Frozen source inventory was incomplete')
    frozen = {}
    expected_by_project = {name: set() for name in mappings}
    with open(inventory['manifest'], encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            top = item['path'].split('/', 1)[0]
            if top in mappings:
                if item['type'] not in ('file', 'directory') or item.get('changed_during_hash'):
                    raise RuntimeError('Unresolved source item: ' + item['path'])
                expected_by_project[top].add(item['path'])
                if item['type'] == 'file':
                    frozen[item['path']] = item
    for name, expected in expected_by_project.items():
        actual = set()
        source_fs = Path(fs_path(ROOT / name))
        for current, dirs, files in os.walk(source_fs, followlinks=False, onerror=lambda e: (_ for _ in ()).throw(e)):
            parent = Path(current)
            rel_parent = parent.relative_to(source_fs)
            actual.add((Path(name) / rel_parent).as_posix())
            for child in dirs + files:
                path = parent / child
                rel = (Path(name) / rel_parent / child).as_posix()
                info = path.stat(follow_symlinks=False)
                if getattr(info, 'st_file_attributes', 0) & 0x400 or path.is_symlink():
                    raise RuntimeError('Unexpected source reparse point: ' + rel)
                actual.add(rel)
                if rel in frozen and (info.st_size, info.st_mtime_ns) != (frozen[rel]['size'], frozen[rel]['mtime_ns']):
                    raise RuntimeError('Legacy original changed since archive; review before backup: ' + rel)
        if actual != expected:
            raise RuntimeError(f'Legacy original paths changed: {name}')
        print('Preflight legacy source:', name, 'paths:', len(actual), flush=True)

    excluded_exact = {'archives/restore-checks', 'archives/reports/full-backup-preflight', '.wbt', PARTIAL.name, FINAL.name,
                      REPORT.relative_to(ROOT).as_posix()}
    started = datetime.now(timezone.utc).isoformat()
    records, links, empty_dirs, exclusions = [], [], [], []
    byte_count = 0
    root_fs = Path(fs_path(ROOT))

    def skip(rel):
        if rel in excluded_exact:
            return True
        return '/' not in rel and rel in mappings

    def walk(path, zip_out):
        nonlocal byte_count
        relative = path.relative_to(root_fs).as_posix()
        if skip(relative):
            exclusions.append(relative)
            return
        info = path.stat(follow_symlinks=False)
        attrs = getattr(info, 'st_file_attributes', 0)
        if attrs & 0x400 or path.is_symlink():
            target = os.readlink(path)
            # The raw target and tag are retained; restoration requires checking
            # and rebasing into the new root rather than following it here.
            links.append({'path': relative, 'target': target,
                          'tag': getattr(info, 'st_reparse_tag', None)})
            return
        if path.is_dir():
            with os.scandir(path) as it:
                children = sorted((entry.name for entry in it), key=str.casefold)
            if not children:
                name = 'master/' + relative.rstrip('/') + '/'
                zi = zipfile.ZipInfo.from_file(str(path), arcname=name, strict_timestamps=False)
                zip_out.writestr(zi, b'')
                empty_dirs.append(relative)
            for name in children:
                walk(path / name, zip_out)
            return
        if not path.is_file():
            raise RuntimeError(f'Unsupported file type: {relative}')
        arcname = 'master/' + relative
        zi = zipfile.ZipInfo.from_file(str(path), arcname=arcname, strict_timestamps=False)
        zi.compress_type = zipfile.ZIP_STORED if path.suffix.lower() in STORE_EXTENSIONS else zipfile.ZIP_DEFLATED
        digest = hashlib.sha256()
        with open(path, 'rb') as src, zip_out.open(zi, 'w', force_zip64=True) as dst:
            for block in iter(lambda: src.read(4 * 1024 * 1024), b''):
                digest.update(block)
                dst.write(block)
        after = path.stat(follow_symlinks=False)
        if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'File changed during backup: {relative}')
        # Verify each archived legacy copy against the frozen source hash while
        # creating the ZIP; reject changed copies rather than trusting an old report.
        prefix = 'archives/legacy-projects/'
        if relative.startswith(prefix):
            original_path = relative[len(prefix):]
            expected = frozen.get(original_path)
            if not expected or digest.hexdigest() != expected['sha256']:
                raise RuntimeError('Archived legacy content differs from frozen source: ' + relative)
        records.append({'path': relative, 'size': info.st_size, 'mtime_ns': info.st_mtime_ns,
                        'attributes': attrs, 'sha256': digest.hexdigest()})
        byte_count += info.st_size
        if len(records) % 5000 == 0:
            print(f'Packed {len(records)} files, {byte_count / 1024**3:.2f} GiB source', flush=True)

    try:
        with zipfile.ZipFile(PARTIAL, 'x', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1, allowZip64=True, strict_timestamps=False) as archive:
            with os.scandir(root_fs) as it:
                children = sorted((entry.name for entry in it), key=str.casefold)
            for name in children:
                walk(root_fs / name, archive)
            manifest = {'created_utc': started, 'source_root': str(ROOT),
                        'description': 'Unencrypted ZIP. Verified legacy archive copies replace duplicate original root directories.',
                        'legacy_original_to_archive': mappings, 'excluded': exclusions,
                        'reparse_points': links, 'empty_directories': empty_dirs,
                        'files': records, 'off_device_backup_verified': False,
                        'restore_notes': 'Extract to a separate root. Inspect and rebase reparse points using this manifest; do not link back to the live original. Recreate legacy root locations from the mapped archive copies only if required. Git worktrees, virtual environments and MLflow artifact URIs require independent restore tests.'}
            archive.writestr('BACKUP_MANIFEST.json', json.dumps(manifest, indent=2).encode('utf-8'),
                             compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
            archive.writestr('READ_FIRST.txt', ('FULL RESEARCH DATA BACKUP - NOT ENCRYPTED\n\n'
                'Legacy originals are stored once as master/archives/legacy-projects/<name>.\n'
                'Both active projects, source skills, Git metadata, videos and tracker data are included.\n'
                'Reparse points are recorded in BACKUP_MANIFEST.json and must be rebased on restore.\n'
                'The archive is local until copied to another drive and restore-tested there.\n').encode('utf-8'))

        print(f'Packed {len(records)} files; verifying ZIP and SHA-256...', flush=True)
        with zipfile.ZipFile(PARTIAL, 'r') as archive:
            for i, entry in enumerate(records, 1):
                digest = hashlib.sha256()
                with archive.open('master/' + entry['path']) as stream:
                    for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                        digest.update(block)
                if digest.hexdigest() != entry['sha256']:
                    raise RuntimeError('ZIP hash mismatch: ' + entry['path'])
                if i % 10000 == 0:
                    print(f'Verified {i}/{len(records)} files', flush=True)
        # Both are single-file paths within ROOT; no source directory is moved/deleted.
        if not PARTIAL.resolve().is_relative_to(ROOT.resolve()) or not FINAL.resolve().is_relative_to(ROOT.resolve()):
            raise RuntimeError('Backup output escaped workspace')
        os.replace(PARTIAL, FINAL)
        report = {'status': 'zip_contents_verified', 'zip': str(FINAL), 'files_verified': len(records),
                  'source_bytes': byte_count, 'zip_bytes': FINAL.stat().st_size, 'zip_sha256': sha_file(FINAL),
                  'links_recorded': len(links), 'excluded': exclusions, 'off_device_backup_verified': False,
                  'encrypted': False, 'started_utc': started,
                  'finished_utc': datetime.now(timezone.utc).isoformat()}
        REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
    except Exception as exc:
        REPORT.write_text(json.dumps({'status': 'incomplete', 'error': str(exc), 'files_packed': len(records),
                                      'partial_retained': PARTIAL.exists(), 'started_utc': started}, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
