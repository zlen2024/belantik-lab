"""Repeated-development active correlation, frozen against an existing probe."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import sys
import time

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.build_wikitext2_joint_context_data import load
from scripts.evaluate_swords_external_candidates import DATA, ISING
from word_energy.contextual_substitution import ising_active_correlations, ising_active_uniform_context_control
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri
from word_energy.wikitext_four_spin_wic import STATES, unary

SOURCE = '6bfbcadfb039410dbc206f963b50d2bf'


def auc(scores, labels):
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        raise ValueError('non-informative target in frozen informative subset')
    return float(np.mean((pos[:, None] > neg) + .5*(pos[:, None] == neg)))


def summarize(rows, models):
    groups = sorted({row['source'] for row in rows})
    indexes = {g: np.flatnonzero([r['source'] == g for r in rows]) for g in groups}
    draws = np.random.default_rng(23140).integers(len(groups), size=(1000, len(groups)))
    mean_auc = {name: float(np.mean([[r['scores'][str(seed)][name] for r in rows] for seed in (17,29,43)])) for name in models}
    effects = {}
    for name in ('vector_contextual', 'uniform_context'):
        delta = np.array([[r['scores'][str(seed)]['active_r']-r['scores'][str(seed)][name] for r in rows]
                          for seed in (17,29,43)])
        averaged = delta.mean(axis=0)
        boot = [averaged[np.concatenate([indexes[groups[i]] for i in draw])].mean() for draw in draws]
        effects[name] = {'mean_difference': float(averaged.mean()), 'seed_difference': delta.mean(axis=1).tolist(),
                         'ci95': np.percentile(boot, (2.5, 97.5)).tolist(), 'valid_draws': len(boot)}
    return {'targets': len(rows), 'source_counts': dict(Counter(r['source'] for r in rows)),
            'mean_auc': mean_auc, 'effects': effects}


def main():
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    root = artifact_dir(client, SOURCE)
    source = json.loads((root/'summary.json').read_text(encoding='utf8'))
    saved = json.loads((root/'input_manifest.json').read_text(encoding='utf8'))
    manifest = saved['items']
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('utf8')).hexdigest()
    if digest != saved['manifest_sha256'] or digest != source['manifest_sha256']:
        raise ValueError('input manifest changed')
    predictions = {r['target_id']: r for r in json.loads((root/'label_blind_predictions.json').read_text(encoding='utf8'))}
    historical = {r['target_id']: r for r in json.loads((root/'target_results.json').read_text(encoding='utf8'))}
    if len(manifest) != 219 or len(predictions) != 219 or len(historical) != 132:
        raise ValueError('source subset changed')
    _, metadata = load(DATA)
    params = {}
    for seed, run_id in ISING.items():
        directory = artifact_dir(client, run_id)
        config = json.loads((directory/'config.json').read_text(encoding='utf8'))
        if config['dataset_hashes'] != metadata['hashes']:
            raise ValueError('dataset mismatch')
        with np.load(directory/'results/checkpoint.npz', allow_pickle=False) as checkpoint:
            params[seed] = dict(checkpoint)
        if {k: sha_array(v) for k,v in params[seed].items()} != config['checkpoint_hashes']:
            raise ValueError('checkpoint hash mismatch')
    rows, full_predictions = [], []
    start = time.monotonic()
    for item in manifest:
        target_id, candidate_ids = item['target_id'], np.asarray(item['candidate_ids'], dtype=np.int64)
        target, context = item['target'], item['context']
        output = {'target_id': target_id, 'source': item['source'], 'seeds': {}}
        for seed in (17, 29, 43):
            model = params[seed]
            zero = dict(model, interaction=np.full(4, -1000.))
            values = {'active_r': ising_active_correlations(model, target, context, candidate_ids),
                      'uniform_context': ising_active_uniform_context_control(model, target, len(context), candidate_ids),
                      'zero_coupling': ising_active_correlations(zero, target, context, candidate_ids)}
            q = unary(model)
            means = q @ STATES
            np.testing.assert_allclose(values['zero_coupling'], means[candidate_ids] @ means[target]/4, atol=1e-12, rtol=0)
            edge = (STATES*(1.5*expit(model['interaction'])/(len(context)+1))) @ STATES.T
            weights = q[target][None,:,None]*q[candidate_ids][:,None,:]*np.exp(edge)[None,:,:]
            expected = np.sum(weights*((STATES @ STATES.T)/4)[None,:,:], axis=(1,2))/np.sum(weights, axis=(1,2))
            np.testing.assert_allclose(values['uniform_context'], expected, atol=1e-12, rtol=0)
            if any(not np.isfinite(v).all() or np.any(np.abs(v) > 1+1e-12) for v in values.values()):
                raise FloatingPointError('invalid active correlation')
            output['seeds'][str(seed)] = {key: value.tolist() for key,value in values.items()}
        full_predictions.append(output)
        if target_id not in historical:
            continue
        prior = historical[target_id]
        if prior['source'] != item['source'] or len(prior['true_fraction']) != len(candidate_ids):
            raise ValueError('candidate source alignment mismatch')
        fractions = prior['true_fraction']
        mask = np.array([fraction is not None for fraction in fractions])
        labels = np.array([fraction is not None and fraction > .5 for fraction in fractions])[mask]
        vector = predictions[target_id]
        row = {'target_id': target_id, 'source': item['source'], 'scores': {}}
        for seed in (17,29,43):
            new = output['seeds'][str(seed)]
            old = vector['scores'][str(seed)]
            row['scores'][str(seed)] = {name: auc(np.asarray(values)[mask], labels)
                for name, values in {**new, 'vector_contextual': old['vector_contextual'],
                                     'affinity': old['ising_affinity'], 'frequency': old['frequency']}.items()}
        rows.append(row)
    if len(rows) != 132:
        raise ValueError('informative target count changed')
    names = ('active_r', 'uniform_context', 'zero_coupling', 'vector_contextual', 'affinity', 'frequency')
    all_results = summarize(rows, names)
    subset = [r for r in rows if r['source'] != 'Nathans_Bylichka.txt']
    sensitivity = summarize(subset, names)
    output = {'source_run': SOURCE, 'manifest_sha256': digest, 'source_artifact_sha256': {
        name: sha256_file(root/name) for name in ('input_manifest.json', 'label_blind_predictions.json', 'target_results.json')},
        'data_run': DATA, 'checkpoints': ISING, 'all': all_results, 'without_dominant_source': sensitivity,
        'elapsed_seconds': time.monotonic()-start,
        'limitations': 'repeated posthoc SWORDS development subset; candidate-specific stars; raw R not energy/probability; CPU exact; no Gaussian or new THRML'}
    mlflow.set_experiment('word-energy-natural-context-evaluation')
    with mlflow.start_run(run_name='swords-posthoc-active-window-R') as run:
        log_contract(mlflow, RunContract('classical', 'cpu_gpu_simulator', 'swords_active_window_correlation',
            'R=mean shared-window target-candidate spin product', '4-spin binary word states', 'beta=1',
            'exact candidate-specific star factorization', 23140, digest,
            'reused SWORDS development, official test closed', '4*(m+2) spins, 10m+16 edges', output['limitations']), output)
        mlflow.log_dict(output, 'summary.json')
        mlflow.log_dict(rows, 'target_results.json')
        mlflow.log_dict(full_predictions, 'predictions.json')
        for path in (Path(__file__), ROOT/'notes/SWORDS_ACTIVE_R_PROBE_PROTOCOL.md',
                     ROOT/'word_energy/contextual_substitution.py', ROOT/'tests/test_contextual_substitution.py'):
            mlflow.log_artifact(str(path), 'source_snapshot')
        print('RUN', run.info.run_id, json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
