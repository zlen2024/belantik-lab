"""Paired article bootstrap for the three-seed full-softmax offset ablation."""
from pathlib import Path
import json
import sys

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from word_energy.tracking import tracking_uri

mlflow.set_tracking_uri(tracking_uri())
client = MlflowClient()
experiment = client.get_experiment_by_name('word-energy-full-softmax-prior-ablation')
runs = client.search_runs([experiment.experiment_id], max_results=100)
selected = {}
for seed in (17, 29, 43):
    candidates = [r for r in runs if r.data.tags.get('seed') == str(seed) and r.info.status == 'FINISHED']
    if len(candidates) != 1:
        raise ValueError(f'expected one completed run for seed {seed}, found {len(candidates)}')
    selected[seed] = candidates[0]
results = {}
for panel in ('wiki', 'ewt'):
    seed_values = []
    for seed, run in selected.items():
        with np.load(artifact_dir(client, run.info.run_id)/'results/ranks.npz') as data:
            docs = data[panel+'_docs']
            doc_ids = np.unique(docs)
            values = {key: np.array([(1/data[panel+'_'+key][docs == d]).mean() for d in doc_ids])
                      for key in ('new', 'old', 'old_corrected', 'margin', 'prior')}
            reference = {'wiki': '782220fbb92b4f18ac410c53e0e44ff4',
                         'ewt': '6219acde7e58415ba42ba9b067f7061e'}[panel]
            with np.load(artifact_dir(client, reference)/'ranks/ranks.npz') as historical:
                if not np.array_equal(docs, historical['doc']):
                    raise ValueError('vector reference document mismatch')
                vector = historical[f'vector10_{seed}']
                values['vector'] = np.array([(1/vector[docs == d]).mean() for d in doc_ids])
            seed_values.append(values)
    results[panel] = {'documents': len(doc_ids), 'mean_mrr': {}, 'comparisons': {}}
    for key in ('new', 'old', 'old_corrected', 'margin', 'prior', 'vector'):
        results[panel]['mean_mrr'][key] = float(np.mean([row[key].mean() for row in seed_values]))
    for key in ('old', 'old_corrected', 'margin', 'prior', 'vector'):
        differences = np.array([row['new'] - row[key] for row in seed_values])
        pooled = differences.mean(axis=0)
        draws = np.random.default_rng(23140).integers(len(pooled), size=(1000, len(pooled)))
        ci = np.percentile(pooled[draws].mean(axis=1), [2.5, 97.5])
        results[panel]['comparisons'][key] = {'delta': float(pooled.mean()), 'ci95': ci.tolist(),
            'seed_delta': differences.mean(axis=1).tolist(),
            'advance': bool(ci[0] > 0 and np.all(differences.mean(axis=1) > 0))}
mlflow.set_experiment('word-energy-full-softmax-prior-ablation')
with mlflow.start_run(run_name='three-seed-summary') as run:
    mlflow.log_dict({'results': results, 'runs': {str(k): v.info.run_id for k,v in selected.items()},
        'limitations': 'reused development panels, exploratory, CPU exact and no physical hardware'}, 'summary.json')
    mlflow.log_artifact(str(Path(__file__)), 'source_snapshot')
    for panel in results:
        mlflow.log_metric(panel+'.delta_vs_old', results[panel]['comparisons']['old']['delta'])
    print('SUMMARY', run.info.run_id, json.dumps(results), flush=True)
