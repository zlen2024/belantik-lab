"""Summarize matched full-softmax and margin development runs."""
from pathlib import Path
import json
import sys
import tempfile

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from word_energy.tracking import tracking_uri

mlflow.set_tracking_uri(tracking_uri())
client = MlflowClient()
experiment = client.get_experiment_by_name('word-energy-full-softmax-pilot')
runs = client.search_runs([experiment.experiment_id], max_results=100)
seeds = (17, 29, 43)
selected = {}
for seed in seeds:
    candidates = [r for r in runs if r.data.tags.get('seed') == str(seed) and r.info.status == 'FINISHED']
    if len(candidates) != 1:
        raise ValueError(f'expected one completed run for seed {seed}, found {len(candidates)}')
    selected[seed] = candidates[0]
results = {}
for panel in ('wiki', 'ewt'):
    effects, diagnostics = [], []
    for seed, run in selected.items():
        with np.load(artifact_dir(client, run.info.run_id)/'results/ranks.npz') as data:
            docs = data[panel+'_docs']
            document_ids = np.unique(docs)
            keys = ('raw', 'corrected', 'margin_raw', 'baseline')
            values = {k: np.array([(1/data[panel+'_'+k][docs == d]).mean() for d in document_ids]) for k in keys}
            effects.append(values['raw'] - values['baseline'])
            diagnostics.append({k: float(v.mean()) for k, v in values.items()})
    effects = np.array(effects)
    pooled = effects.mean(axis=0)
    draw = np.random.default_rng(23140).integers(len(pooled), size=(1000, len(pooled)))
    ci = np.percentile(pooled[draw].mean(axis=1), [2.5, 97.5])
    results[panel] = {'delta': float(pooled.mean()), 'ci95': ci.tolist(),
                      'seed_delta': effects.mean(axis=1).tolist(), 'article_count': len(pooled),
                      'advance': bool(ci[0] > 0 and np.all(effects.mean(axis=1) > 0)),
                      'readout_diagnostics': diagnostics}
mlflow.set_experiment('word-energy-full-softmax-pilot')
with mlflow.start_run(run_name='three-seed-summary') as run:
    mlflow.log_dict({'results': results, 'runs': {str(k): v.info.run_id for k,v in selected.items()},
                     'limitations': 'development panels reused; objective/readout pipeline comparison; unequal compute; classical exact'}, 'summary.json')
    mlflow.log_artifact(str(Path(__file__)), 'source_snapshot')
    for panel, result in results.items():
        mlflow.log_metric(panel+'.article_mrr_delta', result['delta'])
    print('SUMMARY', run.info.run_id, json.dumps(results), flush=True)
