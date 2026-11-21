"""Audit information and shortcut properties of a four-token bridge candidate."""
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
import hashlib
import json
import math
import sys

import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from word_energy.tracking import RunContract, log_contract, tracking_uri


def conditional_entropy(rows, fields):
    groups = defaultdict(Counter)
    for row in rows:
        groups[tuple(row[field] for field in fields)][row['label']] += 1
    n = len(rows)
    value = 0.
    for counts in groups.values():
        size = sum(counts.values())
        for count in counts.values():
            p = count/size
            value -= size/n*p*math.log2(p)
    return value


def main():
    split_rows = {'train': [], 'development': [], 'reserved': []}
    split_names = ('train', 'train', 'development', 'reserved')
    for a, b, ia, ib, target in product(range(2), range(2), range(4), range(4), range(2)):
        y = a ^ b ^ target
        row = {'target': f'T{target}', 'a': f'A{a}_{ia}', 'b': f'B{b}_{ib}',
               'a_family': a, 'b_family': b, 'target_type': target, 'label': y,
               'positive_anchor': f'S{y}', 'negative_anchor': f'S{1-y}'}
        split_rows[split_names[(ia+ib) % 4]].append(row)
    audits = {}
    pair_sets = {split: {(row['a'], row['b']) for row in rows} for split, rows in split_rows.items()}
    for first, second in (('train', 'development'), ('train', 'reserved'), ('development', 'reserved')):
        if pair_sets[first] & pair_sets[second]:
            raise ValueError('pair splits overlap')
    manifest = {}
    feature_sets = {'none': (), 'target': ('target',), 'cue_a': ('a',), 'cue_b': ('b',),
                    'target_a': ('target', 'a'), 'target_b': ('target', 'b'), 'a_b': ('a', 'b'),
                    'target_a_b': ('target', 'a', 'b'), 'families': ('target_type', 'a_family', 'b_family')}
    for split, rows in split_rows.items():
        cue_ids = {row['a'] for row in rows} | {row['b'] for row in rows}
        target_ids = {row['target'] for row in rows}
        anchors = {row['positive_anchor'] for row in rows}
        cells = Counter((row['target_type'], row['a_family'], row['b_family']) for row in rows)
        if len(cue_ids) != 16 or len(target_ids) != 2 or len(anchors) != 2 or len(cells) != 8 or len(set(cells.values())) != 1:
            raise ValueError('coverage or balance failure')
        entropy = {name: conditional_entropy(rows, fields) for name, fields in feature_sets.items()}
        for key in ('none', 'target', 'cue_a', 'cue_b', 'target_a', 'target_b', 'a_b'):
            if abs(entropy[key]-1.) > 1e-12:
                raise ValueError(f'{split} information shortcut for {key}')
        for key in ('target_a_b', 'families'):
            if abs(entropy[key]) > 1e-12:
                raise ValueError('full input does not determine synthetic label')
        positives = {(r['positive_anchor'], r['target'], r['a'], r['b']) for r in rows}
        negatives = {(r['negative_anchor'], r['target'], r['a'], r['b']) for r in rows}
        collisions = len(positives & negatives)
        if collisions:
            raise ValueError('positive-negative collision')
        center_counts = Counter((r[k], label) for r in rows for k, label in (('positive_anchor', 1), ('negative_anchor', 0)))
        if any(center_counts[(center, 1)] != center_counts[(center, 0)] for center in anchors):
            raise ValueError('anchor center frequency shortcut')
        audits[split] = {'examples': len(rows), 'unique_cue_pairs': len(pair_sets[split]),
                         'cue_ids': len(cue_ids), 'cells': len(cells), 'per_cell': next(iter(cells.values())),
                         'label_entropy_bits': entropy, 'center_only_accuracy': .5, 'positive_negative_collisions': collisions}
        if split != 'reserved':
            manifest[split] = rows
    encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('utf8')
    output = {'splits': audits, 'manifest_sha256': hashlib.sha256(encoded).hexdigest(),
              'limitations': 'synthetic supervised anchor observations, deterministic split, no model or chip evaluation'}
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment('word-energy-two-cue-fixtures')
    with mlflow.start_run(run_name='two-cue-four-token-bridge-data-audit') as run:
        log_contract(mlflow, RunContract('classical', 'cpu_gpu_simulator', 'two_cue_bridge_information_audit',
            'no energy model scored', 'synthetic symbolic token IDs', 'not applicable',
            'deterministic combinatorial enumeration', 0, output['manifest_sha256'],
            'synthetic train/development, reserved aggregate only', 'prospective four-token graph not implemented', output['limitations']), output)
        mlflow.log_dict(output, 'audit.json')
        mlflow.log_dict(manifest, 'train_development_manifest.json')
        for path in (Path(__file__), ROOT/'notes/TWO_CUE_BRIDGE_AUDIT_PROTOCOL.md'):
            mlflow.log_artifact(str(path), 'source_snapshot')
        print('RUN', run.info.run_id, json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
