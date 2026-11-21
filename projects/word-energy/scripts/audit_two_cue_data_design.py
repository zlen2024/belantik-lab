"""Audit two-cue split, supervision and negative-sampling shortcuts."""
from collections import Counter, defaultdict
from itertools import product, permutations
from pathlib import Path
import hashlib
import json
import sys

import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from word_energy.tracking import RunContract, log_contract, tracking_uri


def main():
    pairs = {'train': [], 'development': [], 'reserved': []}
    assignment = ('train', 'train', 'development', 'reserved')
    for a_value, b_value, a_index, b_index in product(range(2), range(2), range(4), range(4)):
        key = (f'A{a_value}_{a_index}', f'B{b_value}_{b_index}')
        split = assignment[(a_index+b_index) % 4]
        pairs[split].append({'a': key[0], 'b': key[1], 'a_value': a_value,
                             'b_value': b_value, 'label': a_value ^ b_value})
    pair_sets = {name: {(row['a'], row['b']) for row in rows} for name, rows in pairs.items()}
    for first, second in (('train', 'development'), ('train', 'reserved'), ('development', 'reserved')):
        if pair_sets[first] & pair_sets[second]:
            raise ValueError('cue pair appears across splits')
    expected_cues = {f'{side}{value}_{index}' for side, value, index in product(('A', 'B'), range(2), range(4))}
    checks = {}
    for split, rows in pairs.items():
        cue_ids = {row['a'] for row in rows} | {row['b'] for row in rows}
        if cue_ids != expected_cues:
            raise ValueError('split does not cover every cue ID')
        cells = Counter((row['a_value'], row['b_value']) for row in rows)
        if len(set(cells.values())) != 1 or len(cells) != 4:
            raise ValueError('split cells are imbalanced')
        by_cue = defaultdict(Counter)
        for row in rows:
            by_cue[row['a']][row['label']] += 1
            by_cue[row['b']][row['label']] += 1
        if any(counts[0] != counts[1] for counts in by_cue.values()):
            raise ValueError('one-cue label distributions are imbalanced')
        checks[split] = {'cue_pairs': len(rows), 'unique_cues': len(cue_ids),
                         'per_cell': {f'{a}{b}': cells[(a, b)] for a, b in product(range(2), repeat=2)},
                         'single_cue_balanced': True}
    train = pairs['train']
    target_inputs = sorted(('T', row['a'], row['b']) for row in train)
    original_target_labels = [row['label'] for row in train]
    swapped_target_labels = [1-row['label'] for row in train]
    assert original_target_labels != swapped_target_labels
    # Target training windows contain no target sense labels; changing those
    # labels leaves the recorded target inputs unchanged, holding anchors fixed.
    anchor_windows = [(f'S{row["label"]}', row['a'], row['b']) for row in train]
    lexical_windows = anchor_windows + target_inputs
    lexical_pairs = {pair for window in lexical_windows for pair in permutations(window, 2)}
    anchor_cue_support = {}
    for family, value in product(('A', 'B'), range(2)):
        cue_names = [f'{family}{value}_{i}' for i in range(4)]
        for anchor in ('S0', 'S1'):
            missing = [cue for cue in cue_names if (anchor, cue) not in lexical_pairs]
            if missing:
                raise ValueError(f'anchor/cue support incomplete: {missing}')
            anchor_cue_support[f'{anchor}_{family}{value}'] = len(cue_names)
    false_anchor_negatives = sum((f'S{1-row["label"]}', row['a']) in lexical_pairs
                                 for row in train)
    target_corruption = [('T', 'positive'), ('D', 'negative')]
    center_shortcut_accuracy = sum((center == 'T') == (label == 'positive')
                                   for center, label in target_corruption)/len(target_corruption)
    assert false_anchor_negatives == len(train) and center_shortcut_accuracy == 1.
    manifest = {'train': train, 'development': pairs['development']}
    encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('utf8')
    output = {'split_audit': checks, 'manifest_sha256': hashlib.sha256(encoded).hexdigest(),
              'train_T_positive_windows': len(target_inputs), 'anchor_positive_windows': len(anchor_windows),
              'target_sense_labels_present_in_inputs': False,
              'target_label_complement_leaves_T_windows_identical': True,
              'scope_of_label_symmetry': 'T inputs alone; anchor windows remain fixed and carry XOR supervision',
              'anchor_cue_support': anchor_cue_support,
              'naive_opposite_anchor_as_lexical_negative_collisions': false_anchor_negatives,
              'proposed_T_vs_never_positive_D_center_shortcut_accuracy': center_shortcut_accuracy,
              'limitations': 'symbolic hand-designed data, no trained model, held-out cue combinations only, no reserved model evaluation'}
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment('word-energy-two-cue-fixtures')
    with mlflow.start_run(run_name='two-cue-data-supervision-audit') as run:
        log_contract(mlflow, RunContract('classical', 'cpu_gpu_simulator', 'two_cue_data_design_audit',
            'no energy model scored', 'synthetic symbolic token IDs', 'not applicable',
            'deterministic combinatorial enumeration', 0, output['manifest_sha256'],
            'synthetic train/development and reserved aggregate only', 'no model graph', output['limitations']), output)
        mlflow.log_dict(manifest, 'train_development_manifest.json')
        mlflow.log_dict(output, 'audit.json')
        for path in (Path(__file__), ROOT/'notes/TWO_CUE_DATA_AUDIT_PROTOCOL.md'):
            mlflow.log_artifact(str(path), 'source_snapshot')
        print('RUN', run.info.run_id, json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
