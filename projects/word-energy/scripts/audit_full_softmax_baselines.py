"""Independently check frozen baseline checkpoint and rowwise rank provenance."""
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
from scripts.build_ewt_retrieval_windows import build_ewt
from scripts.evaluate_four_spin_full_vocab import ranks_for
from word_energy.text8_development import sha_array
from word_energy.tracking import tracking_uri
from word_energy.wikitext_four_spin import initial

mlflow.set_tracking_uri(tracking_uri())
client = MlflowClient()
data, meta = load('f1b7db5231ed4a5788b566e388f8e731')
prior = .25 * np.log(data['counts'].astype(float))
ewt, ewt_doc, ewt_info = build_ewt(ROOT, meta['vocabulary'])
panels = {'wiki': (data['validation_windows'], data['validation_document'], '782220fbb92b4f18ac410c53e0e44ff4'),
          'ewt': (ewt, ewt_doc, '6219acde7e58415ba42ba9b067f7061e')}
summary = json.loads((artifact_dir(client, 'c05a1d631d814a9db373876c83d7d413')/'config.json').read_text('utf8'))
checks = {}
for seed in (17, 29, 43):
    source = summary['results'][f'joint4:{seed}']['run_id']
    directory = artifact_dir(client, source)
    config = json.loads((directory/'config.json').read_text('utf8'))
    assert config['dataset_run_id'] == 'f1b7db5231ed4a5788b566e388f8e731'
    assert config['dataset_hashes'] == meta['hashes']
    assert config['offset_sha256'] == sha_array(prior)
    assert config['checkpoint_hashes']['initial'] == {k: sha_array(np.asarray(v)) for k,v in initial('joint4', seed, len(prior)).items()}
    with np.load(directory/'checkpoints/epoch3.npz', allow_pickle=False) as saved:
        params = dict(saved)
    assert config['checkpoint_hashes']['epoch3'] == {k: sha_array(v) for k,v in params.items()}
    for panel, (windows, docs, ref) in panels.items():
        reference_config = json.loads((artifact_dir(client, ref)/'config.json').read_text('utf8'))
        if panel == 'wiki':
            assert reference_config['hashes'] == meta['hashes']
        else:
            assert reference_config['panel']['windows_sha256'] == ewt_info['windows_sha256']
        ranks, error = ranks_for(params, windows, prior + np.log(data['noise_q']), 'joint4')
        with np.load(artifact_dir(client, ref)/'ranks/ranks.npz') as saved:
            assert np.array_equal(saved['doc'], docs)
            assert np.array_equal(saved[f'joint4_{seed}'], ranks)
        checks[f'{panel}_{seed}'] = {'changed_ranks': 0, 'score_max_error': error, 'source': source}
        print(panel, seed, 'passed', flush=True)
mlflow.set_experiment('word-energy-full-softmax-pilot')
with mlflow.start_run(run_name='baseline-provenance-audit') as run:
    mlflow.log_dict(checks, 'audit.json')
    mlflow.log_artifact(str(Path(__file__)), 'source_snapshot')
    print('AUDIT', run.info.run_id, json.dumps(checks), flush=True)
