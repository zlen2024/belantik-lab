"""Generate a portable archive index from copy reports; never invent outcomes."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT / 'archives'
PROJECTS = ['analog-lab-learn', 'attn_thrml', 'channel', 'DTM_deepdive', 'fundamental_diff',
            'mnist_diff', 'ncm_torx', 'normal_computing', 'p_family_xor',
            'rbm-anomaly-detection', 'thrml', 'thrml_linalg', 'torx',
            'vbuild_assets', 'videos', 'word2energy', 'z1_voters']


def main():
    entries = []
    parity_file = ARCHIVES / 'reports' / 'git_parity.json'
    parity = json.loads(parity_file.read_text(encoding='utf-8')) if parity_file.exists() else []
    for name in PROJECTS:
        reports = ARCHIVES / 'reports'
        retries = sorted(reports.glob(name + '.retry-*.json'))
        report = retries[-1] if retries else reports / (name + '.json')
        copy = ARCHIVES / 'legacy-projects' / name
        details = json.loads(report.read_text(encoding='utf-8')) if report.exists() else {}
        status = details.get('status', 'copy_in_progress' if copy.exists() else 'not_started')
        checks = [row for row in parity if row['repo'].split('/')[0] == name]
        git_status = ('verified_at_check' if checks and all(x['all_pass'] for x in checks)
                      else 'failed_or_incomplete' if checks else 'not_applicable_or_not_checked')
        restore_report = ARCHIVES / 'reports' / 'restore-local-analog.json'
        restored = (json.loads(restore_report.read_text(encoding='utf-8')).get('status')
                    if name == 'analog-lab-learn' and restore_report.exists() else 'not_tested')
        entries.append({'id': 'legacy-' + name.lower(), 'title': name, 'original_path': name,
                        'archive_path': 'legacy-projects/' + name, 'status': status,
                        'content_files_verified': details.get('files_verified'),
                        'git_history_check': git_status,
                        'local_restore_check': restored, 'off_device_backup': 'not_verified',
                        'hypothesis': None, 'outcome': None})
    root_report = ARCHIVES / 'reports' / 'legacy-root-files.json'
    root_details = json.loads(root_report.read_text(encoding='utf-8')) if root_report.exists() else {}
    entries.append({'id': 'legacy-root-files', 'title': 'Legacy root files', 'original_path': '.',
                    'archive_path': 'legacy-root-files', 'status': root_details.get('status', 'not_started'),
                    'content_files_verified': root_details.get('files_verified'),
                    'git_history_check': 'not_applicable', 'off_device_backup': 'not_verified',
                    'hypothesis': None, 'outcome': None})
    # JSON is valid YAML 1.2 and preserves Unicode without requiring a YAML dependency.
    document = {'schema_version': 1, 'note': 'Local archive copies; originals retained. Null = unknown.',
                'source_inventory': 'research_workspace_planning/inventory (private local; not committed)',
                'entries': entries}
    (ARCHIVES / 'index.yaml').write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Archive index updated:', len(entries), 'entries')


if __name__ == '__main__':
    main()
