"""Repeated-development paired check of one shared multi-candidate Ising star."""
from pathlib import Path
import hashlib
import json
import sys
import time

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.build_wikitext2_joint_context_data import load
from scripts.evaluate_swords_external_candidates import DATA, ISING, load_checkpoint
from scripts.evaluate_swords_active_r import auc
from word_energy.contextual_substitution import ising_shared_correlations
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri
from word_energy.wikitext_four_spin_wic import STATES, unary

ORIGINAL = '6bfbcadfb039410dbc206f963b50d2bf'
ACTIVE = '3878d013f4ae477f8af4c1e41cf1252b'
SEEDS = (17, 29, 43)


def evaluate_summary(rows):
    groups = sorted({r['source'] for r in rows})
    idx = {g: np.flatnonzero([r['source'] == g for r in rows]) for g in groups}
    draws = np.random.default_rng(23140).integers(len(groups), size=(1000, len(groups)))
    names = ('shared', 'uniform', 'active_r', 'vector_contextual')
    result = {'targets': len(rows), 'source_groups': len(groups), 'models': {}, 'effects': {}}
    for name in names:
        vals = np.array([[r['scores'][str(seed)][name] for r in rows] for seed in SEEDS])
        result['models'][name] = {'mean': float(vals.mean()), 'per_seed': vals.mean(axis=1).tolist()}
    for other in names[1:]:
        delta = np.array([[r['scores'][str(seed)]['shared']-r['scores'][str(seed)][other]
                           for r in rows] for seed in SEEDS])
        averaged = delta.mean(axis=0)
        boot = [averaged[np.concatenate([idx[groups[i]] for i in draw])].mean() for draw in draws]
        result['effects'][other] = {'mean_difference': float(delta.mean()),
            'per_seed': delta.mean(axis=1).tolist(), 'ci95': np.percentile(boot, (2.5, 97.5)).tolist()}
    return result


def main():
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    origin, active = artifact_dir(client, ORIGINAL), artifact_dir(client, ACTIVE)
    saved = json.loads((origin/'input_manifest.json').read_text(encoding='utf8'))
    manifest = saved['items']
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('utf8')).hexdigest()
    if digest != saved['manifest_sha256'] or len(manifest) != 219:
        raise ValueError('input manifest mismatch')
    _, metadata = load(DATA)
    models = {seed: load_checkpoint(client, ISING[seed], metadata) for seed in SEEDS}
    predictions = []
    started = time.monotonic()
    for item in manifest:
        ids = np.array(item['candidate_ids'], dtype=np.int64)
        record = {'target_id': item['target_id'], 'source': item['source'], 'seeds': {}}
        for seed, params in models.items():
            expanded = dict(params)
            expanded['biases'] = np.vstack((params['biases'], np.zeros((1, 4))))
            expanded['internal'] = np.vstack((params['internal'], np.zeros((1, 6))))
            dummy = [len(params['biases'])]*len(item['context'])
            values = ising_shared_correlations(params, item['target'], item['context'], ids)
            control = ising_shared_correlations(expanded, item['target'], dummy, ids)
            zero = dict(params, interaction=np.full(4, -1000.))
            zero_scores = ising_shared_correlations(zero, item['target'], item['context'], ids)
            means = unary(params) @ STATES
            np.testing.assert_allclose(zero_scores, means[ids] @ means[item['target']]/4, atol=1e-12, rtol=0)
            if not np.isfinite(values).all() or not np.isfinite(control).all() or np.any(np.abs(values)>1+1e-12) or np.any(np.abs(control)>1+1e-12):
                raise FloatingPointError('invalid correlation')
            record['seeds'][str(seed)] = {'shared': values.tolist(), 'uniform': control.tolist()}
        predictions.append(record)

    # The stored labels are opened only after all new predictions have been produced.
    historical = {r['target_id']: r for r in json.loads((origin/'target_results.json').read_text(encoding='utf8'))}
    old_predictions = {r['target_id']: r for r in json.loads((origin/'label_blind_predictions.json').read_text(encoding='utf8'))}
    active_predictions = {r['target_id']: r for r in json.loads((active/'predictions.json').read_text(encoding='utf8'))}
    rows, rank_changes = [], []
    for item, prediction in zip(manifest, predictions):
        tid = item['target_id']
        if tid not in historical:
            continue
        prior = historical[tid]
        fractions = prior['true_fraction']
        if prior['source'] != item['source'] or len(fractions) != len(item['candidate_ids']):
            raise ValueError('historical input alignment mismatch')
        mask = np.array([v is not None for v in fractions])
        labels = np.array([v is not None and v > .5 for v in fractions])[mask]
        row = {'target_id': tid, 'source': item['source'], 'scores': {}}
        for seed in SEEDS:
            key = str(seed)
            scores = dict(prediction['seeds'][key])
            scores['active_r'] = active_predictions[tid]['seeds'][key]['active_r']
            scores['vector_contextual'] = old_predictions[tid]['scores'][key]['vector_contextual']
            row['scores'][key] = {name: auc(np.asarray(values)[mask], labels) for name,values in scores.items()}
            a, b = np.asarray(scores['shared'])[mask], np.asarray(scores['active_r'])[mask]
            upper = np.triu_indices(len(a), 1)
            rank_changes.append(float(np.mean(np.sign(a[:,None]-a[None,:])[upper] != np.sign(b[:,None]-b[None,:])[upper])))
        rows.append(row)
    if len(rows) != 132:
        raise ValueError('informative subset mismatch')
    result = {'source_run': ORIGINAL, 'active_r_run': ACTIVE, 'manifest_sha256': digest,
              'source_hashes': {f'{label}/{name}': sha256_file(root/name) for label,root,names in
                  [('original',origin,('input_manifest.json','target_results.json','label_blind_predictions.json')),
                   ('active',active,('predictions.json',))] for name in names},
              'checkpoints': ISING, 'all': evaluate_summary(rows),
              'without_dominant_source': evaluate_summary([r for r in rows if r['source'] != 'Nathans_Bylichka.txt']),
              'mean_rank_change_fraction_vs_active': float(np.mean(rank_changes)),
              'elapsed_seconds': time.monotonic()-started,
              'limitations': 'Repeated SWORDS development, already inspected labels; frozen checkpoints trained with different star; candidate count changes coupling; no training, Gaussian, THRML or hardware.'}
    mlflow.set_experiment('word-energy-natural-context-evaluation')
    with mlflow.start_run(run_name='swords-shared-star-posthoc') as run:
        log_contract(mlflow, RunContract('classical','cpu_gpu_simulator','shared-star Ising exact correlation',
            'higher raw mean spin correlation ranks higher','4 spins {-1,+1} per word','beta=1',
            'exact shared-star factorization',23140,digest,'reused SWORDS development; official test closed',
            '4(1+m+K) spins; max degree 3+m+K',result['limitations']), result)
        mlflow.log_dict(result,'summary.json')
        mlflow.log_dict(rows,'target_results.json')
        mlflow.log_dict(predictions,'predictions.json')
        for name, metrics in result['all']['models'].items():
            mlflow.log_metric(name+'.target_macro_auc', metrics['mean'])
        for path in (Path(__file__), ROOT/'notes/SWORDS_SHARED_STAR_PROBE_PROTOCOL.md', ROOT/'word_energy/contextual_substitution.py'):
            mlflow.log_artifact(str(path),'source_snapshot')
        print('RUN',run.info.run_id,json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
