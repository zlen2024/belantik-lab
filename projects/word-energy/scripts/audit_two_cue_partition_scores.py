"""Posthoc joint and additive word-energy scores for both oracle fixtures."""
from pathlib import Path
import json
import sys

import mlflow
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from word_energy.tracking import tracking_uri
from word_energy.wikitext_four_spin import score_windows

mlflow.set_tracking_uri(tracking_uri())
client = mlflow.tracking.MlflowClient()
fixtures = {'flat': 'c61f719ebfe342f49a84fad256ff11aa', 'anchored': '1808b653756e4c1b9b3e7c23f51ac8a5'}
windows = np.array([[0, 1, 3], [0, 1, 4], [0, 2, 3], [0, 2, 4]])
scores = {}
for name, run_id in fixtures.items():
    artifact = client.download_artifacts(run_id, 'fixture.json')
    with open(artifact, encoding='utf8') as f:
        fixture = json.load(f)
    params = {k: np.asarray(v) for k, v in fixture['parameters'].items()}
    zero_offset = np.zeros(params['biases'].shape[0])
    scores[name] = {arm: np.asarray(score_windows(params, windows, zero_offset, arm)).tolist()
                    for arm in ('joint4', 'pair_add4')}
experiment = client.get_experiment_by_name('word-energy-two-cue-fixtures')
with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='posthoc-fixture-partition-readout') as run:
    mlflow.log_dict({'scores': scores, 'source_runs': fixtures,
                     'limitations': 'hand-set fixture; posthoc exact host scoring, no training or physical chip'}, 'scores.json')
    mlflow.log_artifact(str(Path(__file__)), 'source_snapshot')
    print('RUN', run.info.run_id, json.dumps(scores), flush=True)
