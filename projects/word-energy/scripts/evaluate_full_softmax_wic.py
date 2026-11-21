"""Repeat the historical WiC development readout on matched full-softmax models."""
from pathlib import Path
import json
import sys

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.build_wikitext2_joint_context_data import load
from scripts.evaluate_wic_context_probe import bootstrap
from scripts.evaluate_wikitext_joint_wic import paired_bootstrap
from scripts.run_full_softmax_matched_vector import ISING
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract, log_contract, tracking_uri, sha256_file
from word_energy.wic_context_probe import auc, cosine, negative_js
from word_energy.wikitext_four_spin_wic import star

REFERENCE = 'fa6a014eefc549338b7735a9e52206ed'
HISTORICAL = '1e43fb7e38524085b4807478b0dd503f'
VECTORS = {17: 'ba3d1cfad0214f36ac30c9cf66c43b23',
           29: '282b6343c23b443ea852d7ef4076aafa',
           43: 'f28f6fbb35c848d8b2947c448cb61f2d'}
DATA = 'f1b7db5231ed4a5788b566e388f8e731'


def main():
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    _, meta = load(DATA)
    reference_root = artifact_dir(client, REFERENCE)
    ref = json.loads((reference_root/'config.json').read_text(encoding='utf8'))
    old = json.loads((artifact_dir(client, HISTORICAL)/'config.json').read_text(encoding='utf8'))
    if meta['base_hashes'] != ref['dataset_hashes'] or meta['hashes'] != old['dataset_hashes']:
        raise ValueError('training vocabulary or dataset mismatch')
    if old['wic_reference_run'] != REFERENCE or old['wic_sha256'] != ref['wic_sha256']:
        raise ValueError('historical WiC reference mismatch')
    for rel in ('word_energy/wic_context_probe.py', 'scripts/evaluate_wic_context_probe.py'):
        if sha256_file(ROOT/rel) != sha256_file(reference_root/'source_snapshot'/Path(rel).name):
            raise ValueError('historical helper changed')
    selected, labels = ref['selected'], np.asarray(ref['labels'], dtype=np.int32)
    groups = [row['word'] for row in selected]
    if len(selected) != 167 or int(labels.sum()) != 82 or len(set(groups)) != old['target_count']:
        raise ValueError('WiC subset or label count mismatch')
    predictions = {'new_ising': [], 'new_vector': [], 'new_vector_context': []}
    for seed in (17, 29, 43):
        for label, run_id in (('new_ising', ISING[seed]), ('new_vector', VECTORS[seed])):
            directory = artifact_dir(client, run_id)
            config = json.loads((directory/'config.json').read_text(encoding='utf8'))
            if config['dataset_hashes'] != meta['hashes']:
                raise ValueError('checkpoint data mismatch')
            with np.load(directory/'results/checkpoint.npz', allow_pickle=False) as data:
                params = dict(data)
            if {k: sha_array(v) for k,v in params.items()} != config['checkpoint_hashes']:
                raise ValueError('checkpoint mismatch')
            scores, context_scores = [], []
            for row in selected:
                t, c1, c2 = row['target'], row['context1'], row['context2']
                if label == 'new_ising':
                    scores.append(negative_js(star(params, t, c1), star(params, t, c2)))
                else:
                    v = params['shared']
                    left, right = np.mean(v[c1], axis=0), np.mean(v[c2], axis=0)
                    scores.append(cosine(v[t]+left, v[t]+right))
                    context_scores.append(cosine(left, right))
            if not np.isfinite(scores).all() or (context_scores and not np.isfinite(context_scores).all()):
                raise FloatingPointError('nonfinite WiC scores')
            predictions[label].append(scores)
            if context_scores:
                predictions['new_vector_context'].append(context_scores)
    predictions.update({'old_ising': old['predictions']['joint4:epoch3']['scores'],
                        'old_vector': old['predictions']['vector10:epoch3']['scores'],
                        'jaccard': old['predictions']['two_spin:previous:bag_jaccard']['scores']})
    for name, old_key in (('old_ising', 'joint4:epoch3'), ('old_vector', 'vector10:epoch3')):
        computed = np.mean([auc(labels, np.asarray(s)) for s in predictions[name]])
        if not np.isclose(computed, old['summaries'][old_key]['auc_mean'], atol=1e-14, rtol=0):
            raise ValueError('historical AUC mismatch')
    summaries = {}
    for name, scores in predictions.items():
        arr = [np.asarray(s) for s in scores]
        summaries[name] = {'auc_per_seed': [auc(labels, a) for a in arr],
                           'auc_mean': float(np.mean([auc(labels, a) for a in arr])),
                           'interval': bootstrap(labels, groups, arr),
                           'distinct_scores_per_seed': [int(len(np.unique(a))) for a in arr]}
    effects = {name: paired_bootstrap(labels, groups, predictions['new_ising'], predictions[name])
               for name in ('old_ising', 'new_vector', 'jaccard')}
    output = {'dataset_run': DATA, 'dataset_hashes': meta['hashes'], 'wic_reference': REFERENCE,
              'historical_run': HISTORICAL, 'wic_sha256': ref['wic_sha256'], 'selected': len(selected),
              'total': ref['total_examples'], 'positive': int(labels.sum()), 'negative': int(len(labels)-labels.sum()),
              'target_count': len(set(groups)), 'sources': {'ising': ISING, 'vector': VECTORS},
              'summaries': summaries, 'effects': effects, 'predictions': predictions,
              'limitations': 'reused development data; 167/638 subset; untrained readouts; context length differs from training; no physical hardware'}
    mlflow.set_experiment('word-energy-full-softmax-matched-vector')
    with mlflow.start_run(run_name='full-softmax-repeated-wic-development') as run:
        log_contract(mlflow, RunContract('classical', 'cpu_gpu_simulator', 'full_softmax_repeated_wic',
            'normalized 16-state Ising posterior or tied vector cosine', '4 spins or 10D vector', 'Ising beta=1',
            'exact posterior readout, no WiC training', 22150, ref['wic_sha256'],
            'reused WiC development subset, reserved tests closed', '4-spin center and 1-4 spokes', output['limitations']), output)
        mlflow.log_dict(output, 'wic_probe.json')
        for name in summaries:
            mlflow.log_metric(name+'.auc', summaries[name]['auc_mean'])
        for path in (Path(__file__), ROOT/'notes/FULL_SOFTMAX_WIC_PROBE_PROTOCOL.md',
                     ROOT/'word_energy/wikitext_four_spin_wic.py', ROOT/'word_energy/wic_context_probe.py'):
            mlflow.log_artifact(str(path), 'source_snapshot')
        print('RUN', run.info.run_id, json.dumps({'summaries': summaries, 'effects': effects}), flush=True)


if __name__ == '__main__':
    main()
