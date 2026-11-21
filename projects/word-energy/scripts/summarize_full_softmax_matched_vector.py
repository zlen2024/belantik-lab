"""Summarize matched full-softmax vector versus Ising runs."""
from pathlib import Path
import sys

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.run_full_softmax_matched_vector import ISING
from word_energy.tracking import tracking_uri

mlflow.set_tracking_uri(tracking_uri())
client = MlflowClient()
experiment = client.get_experiment_by_name('word-energy-full-softmax-matched-vector')
runs = client.search_runs([experiment.experiment_id], max_results=100)
selected = {}
for seed in (17, 29, 43):
    candidates = [r for r in runs if r.data.tags.get('seed') == str(seed) and r.info.status == 'FINISHED']
    if len(candidates) != 1:
        raise ValueError(f'expected one finished run for seed {seed}, got {len(candidates)}')
    selected[seed] = candidates[0]
output = {}
for panel in ('wiki', 'ewt'):
    paired, values, auc = [], [], []
    for seed, run in selected.items():
        with np.load(artifact_dir(client, run.info.run_id)/'results/comparison.npz') as ranks:
            docs = ranks[panel+'_docs']
            ids = np.unique(docs)
            doc_scores = {key: np.array([(1/ranks[panel+'_'+key][docs == d]).mean() for d in ids])
                          for key in ('ising', 'vector')}
            with np.load(artifact_dir(client, ISING[seed])/'results/ranks.npz') as previous:
                if not np.array_equal(docs, previous[panel+'_docs']):
                    raise ValueError('frequency reference mismatch')
                prior = previous[panel+'_prior']
                doc_scores['prior'] = np.array([(1/prior[docs == d]).mean() for d in ids])
            paired.append(doc_scores['ising'] - doc_scores['vector'])
            values.append({k: float(v.mean()) for k,v in doc_scores.items()})
            if panel == 'wiki':
                auc.append({k: float(ranks['wiki_auc_'+k].mean()) for k in ('ising', 'vector')})
    paired = np.asarray(paired)
    pooled = paired.mean(axis=0)
    draws = np.random.default_rng(23140).integers(len(ids), size=(1000, len(ids)))
    ci = np.percentile(pooled[draws].mean(axis=1), [2.5, 97.5])
    output[panel] = {'mean_mrr': {k: float(np.mean([row[k] for row in values])) for k in ('ising', 'vector', 'prior')},
        'documents': len(ids), 'ising_minus_vector': float(pooled.mean()), 'ci95': ci.tolist(),
        'seed_deltas': paired.mean(axis=1).tolist(),
        'advance': bool(ci[0] > 0 and np.all(paired.mean(axis=1) > 0)), 'auc_per_seed': auc}
mlflow.set_experiment('word-energy-full-softmax-matched-vector')
with mlflow.start_run(run_name='matched-vector-three-seed-summary') as run:
    mlflow.log_dict({'panels': output, 'runs': {str(k): v.info.run_id for k,v in selected.items()},
                     'ising_runs': ISING, 'limitations': 'reused development panels; conditional article bootstrap; classical CPU'}, 'summary.json')
    mlflow.log_artifact(str(Path(__file__)), 'source_snapshot')
    for panel in output:
        mlflow.log_metric(panel+'.ising_minus_vector', output[panel]['ising_minus_vector'])
    print('SUMMARY', run.info.run_id, output, flush=True)
