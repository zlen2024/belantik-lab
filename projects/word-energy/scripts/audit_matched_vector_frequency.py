"""Descriptive frequency strata for matched Ising and vector noise scores."""
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
from scripts.run_full_softmax_matched_vector import DATA, ISING
from word_energy.text8_lexical_models import pair_metrics
from word_energy.tracking import tracking_uri
from word_energy.wikitext_four_spin import functions
from word_energy.wikitext_joint_evaluation import canonical_scores

mlflow.set_tracking_uri(tracking_uri())
client = MlflowClient()
data, meta = load(DATA)
experiment = client.get_experiment_by_name('word-energy-full-softmax-matched-vector')
runs = client.search_runs([experiment.experiment_id], max_results=100)
offset = np.log(data['counts'].astype(float))
results = {}
for seed in (17, 29, 43):
    matches = [r for r in runs if r.data.tags.get('seed') == str(seed) and r.info.status == 'FINISHED']
    if len(matches) != 1:
        raise ValueError(f'expected one vector run for {seed}')
    for label, run_id, arm in (('ising', ISING[seed], 'joint4'), ('vector', matches[0].info.run_id, 'vector10')):
        directory = artifact_dir(client, run_id)
        config = json.loads((directory/'config.json').read_text('utf8'))
        if config['dataset_hashes'] != meta['hashes']:
            raise ValueError('dataset mismatch')
        with np.load(directory/'results/checkpoint.npz', allow_pickle=False) as checkpoint:
            params = dict(checkpoint)
        positive, negative = canonical_scores(params, functions(arm, offset)[0], data['validation_windows'], data['validation_noise'])
        results[f'{label}_{seed}'] = pair_metrics(positive, negative, data['counts'][data['validation_windows'][:, 0]])
with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='matched-vector-frequency-strata') as run:
    mlflow.log_dict({'results': results, 'dataset_hashes': meta['hashes'],
                     'limitations': 'posthoc window-weighted frequency strata, reused validation and sampled noise'}, 'frequency_strata.json')
    mlflow.log_artifact(str(Path(__file__)), 'source_snapshot')
    print('AUDIT', run.info.run_id, json.dumps(results), flush=True)
