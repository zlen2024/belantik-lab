"""Compare the prior ablation with the historical 20-noise AUC panel."""
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
from scripts.run_full_softmax_prior import DATA
from word_energy.text8_development import sha_array
from word_energy.text8_lexical_models import pair_metrics
from word_energy.tracking import RunContract, log_contract, tracking_uri
from word_energy.wikitext_four_spin import functions
from word_energy.wikitext_joint_evaluation import canonical_scores

NEW = {17: '34ad840584164e5dac2059ca6ef5267b', 29: '6fd4829031424eba917b99c5fd1ab9ca', 43: '2ee7dde861fa4e82b960c0767e4bef6b'}
OLD = {17: '457aaa1ab2af46678844386986a61b32', 29: '1cca3e70769b42a7b8b5e64cf5987b08', 43: 'b1de614b15144b9e98bb2a5e70584b31'}


def load_params(client, run_id, new, metadata):
    directory = artifact_dir(client, run_id)
    config = json.loads((directory/'config.json').read_text(encoding='utf8'))
    if config['dataset_hashes'] != metadata['hashes']:
        raise ValueError('dataset hash mismatch')
    filename = directory/('results/checkpoint.npz' if new else 'checkpoints/epoch3.npz')
    with np.load(filename, allow_pickle=False) as archive:
        params = dict(archive)
    hashes = config['checkpoint_hashes'] if new else config['checkpoint_hashes']['epoch3']
    if {key: sha_array(value) for key, value in params.items()} != hashes:
        raise ValueError('checkpoint hash mismatch')
    return params, config


def main():
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    data, metadata = load(DATA)
    windows, noise, docs = data['validation_windows'], data['validation_noise'], data['validation_document']
    document_ids = np.unique(docs)
    counts = data['counts'].astype(float)
    scorers = {'new': functions('joint4', np.log(counts))[0],
               'old': functions('joint4', .25*np.log(counts))[0],
               'old_full_prior': functions('joint4', np.log(counts))[0]}
    results, per_doc = {}, {}
    for seed in (17, 29, 43):
        new_params, _ = load_params(client, NEW[seed], True, metadata)
        old_params, old_config = load_params(client, OLD[seed], False, metadata)
        for label in scorers:
            params = new_params if label == 'new' else old_params
            positive, negative = canonical_scores(params, scorers[label], windows, noise)
            outcome = ((positive[:, None] > negative) + .5*(positive[:, None] == negative)).mean(axis=1)
            article = np.array([outcome[docs == doc].mean() for doc in document_ids])
            metrics = pair_metrics(positive, negative, counts[windows[:, 0]])
            metrics['document_macro_auc'] = float(article.mean())
            if label == 'old' and not np.isclose(metrics['document_macro_auc'], old_config['metrics']['epoch3']['document_macro_auc'], atol=1e-14, rtol=0):
                raise ValueError('historical AUC did not reproduce')
            results[f'{label}_{seed}'] = metrics
            per_doc[f'{label}_{seed}'] = article
    draws = np.random.default_rng(23140).integers(len(document_ids), size=(1000, len(document_ids)))
    effects = {}
    for label in ('old', 'old_full_prior'):
        differences = np.array([per_doc[f'new_{seed}'] - per_doc[f'{label}_{seed}'] for seed in NEW])
        pooled = differences.mean(axis=0)
        effects[label] = {'delta': float(pooled.mean()),
            'ci95': np.percentile(pooled[draws].mean(axis=1), [2.5, 97.5]).tolist(),
            'seed_deltas': differences.mean(axis=1).tolist()}
    output = {'results': results, 'effects': effects, 'dataset_run': DATA, 'dataset_hashes': metadata['hashes'],
        'new_runs': NEW, 'old_runs': OLD, 'documents': len(document_ids), 'windows': len(windows),
        'noise_per_window': noise.shape[1], 'limitations': 'reused development data, sampled noise, readout differs in primary comparison'}
    mlflow.set_experiment('word-energy-full-softmax-prior-ablation')
    with mlflow.start_run(run_name='paired-historical-noise-auc') as run:
        log_contract(mlflow, RunContract('classical', 'cpu_gpu_simulator', 'joint4_paired_noise_auc',
            'normalized Ising joint score with train-specific center offset', '12 binary spins', 'beta=1',
            'exact scoring of observed and 20 noise candidates', 23140, metadata['hashes']['validation_windows'],
            'WikiText development validation, reserved test closed', '3 K4 and 8 spokes', output['limitations']), output)
        mlflow.log_dict(output, 'auc_comparison.json')
        mlflow.log_artifact(str(Path(__file__)), 'source_snapshot')
        for key, item in results.items():
            mlflow.log_metric(key+'.macro_auc', item['document_macro_auc'])
        print('RUN', run.info.run_id, json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
