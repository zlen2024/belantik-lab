"""Read-only diagnosis of frozen Ising checkpoints on validation windows."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from word_energy.ising.exact_training import _free_energy_factory, _region_mass, _sense_accuracy_and_loss, prepare_data
from word_energy.tracking import sha256_file


def diagnose(seed, checkpoint_kind='ising-exact-training'):
    if checkpoint_kind not in ('ising-exact-training', 'ising-random', 'ising-permuted-lexical'):
        raise ValueError('unknown checkpoint kind')
    checkpoint = ROOT / 'experiments' / 'runs' / f'{checkpoint_kind}-seed{seed}.npz'
    with np.load(checkpoint) as archive:
        raw = {key: archive[key] for key in archive.files}
    _, jnp, energy, unary, center = _free_energy_factory()
    data = prepare_data()
    anchors = {family: [int(token) for row, label in zip(data['train_windows'], data['train_families']) if label == family for token in row[1:]] for family in ('F', 'R')}
    q = {family: np.mean([np.asarray(unary(raw, token)) for token in tokens], axis=0) for family, tokens in anchors.items()}
    region = q['F'] > q['R']
    equal = q['F'] == q['R']
    rows = []
    for index, (window, family) in enumerate(zip(data['validation_windows'], data['validation_families'])):
        if family not in ('F', 'R'):
            continue
        posterior = np.asarray(center(raw, jnp.asarray(window), 1.0))
        prior = np.asarray(center(raw, jnp.asarray(window), 0.0))
        f_mass = _region_mass(posterior, region, equal)
        f_prior = _region_mass(prior, region, equal)
        correct = f_mass if family == 'F' else 1.0 - f_mass
        base = f_prior if family == 'F' else 1.0 - f_prior
        rows.append({'row': index, 'family': family, 'tokens': window.tolist(), 'state_posterior': posterior.tolist(), 'finance_mass': f_mass, 'correct_mass': correct, 'prior_correct_mass': base, 'gain': correct - base, 'credit': 1.0 if correct > .5 else .5 if correct == .5 else 0.0})
    families = {}
    for family in ('F', 'R'):
        group = [r for r in rows if r['family'] == family]
        families[family] = {'count': len(group), **{key: float(np.mean([r[key] for r in group])) for key in ('correct_mass', 'prior_correct_mass', 'gain', 'credit')}}
    accuracy, loss = _sense_accuracy_and_loss(raw, energy, unary, center, data, jnp)
    return {'seed': seed, 'checkpoint_kind': checkpoint_kind, 'checkpoint_sha256': sha256_file(checkpoint), 'protocol_sha256': data['protocol_sha256'], 'state_order': ['--', '+-', '-+', '++'], 'train_anchor_q': {key: value.tolist() for key, value in q.items()}, 'anchor_total_variation': float(np.abs(q['F'] - q['R']).sum() / 2), 'finance_region': region.tolist(), 'equal_region': equal.tolist(), 'bank_prior': np.asarray(unary(raw, data['ids']['bank'])).tolist(), 'family_summary': families, 'validation_macro_sense_accuracy': accuracy, 'validation_joint_ranking_loss': loss, 'validation_rows': rows}


if __name__ == '__main__':
    seed = int(sys.argv[1])
    print(json.dumps(diagnose(seed, sys.argv[2] if len(sys.argv) > 2 else 'ising-exact-training'), indent=2))
