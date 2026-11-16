"""Run a frozen 8-permutation x 4-model-seed exploratory panel."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from word_energy.fresh_synthetic import make_fresh_corpus
from word_energy.ising.exact_training import prepare_data
from word_energy.ising.spectral_initializer import array_hash, spectral_biases
from word_energy.tracking import sha256_file, tracking_uri
from scripts.run_ising_fresh_synthetic import SEEDS
from scripts.run_ising_spectral_ablation import random_initial

PERMUTATIONS = (7011, 7013, 7019, 7027, 7039, 7043, 7057, 7069)
BASELINE = ROOT / 'mlruns' / '8' / 'd4b3e7caa3ee44fb8807f1fc88a23b44' / 'artifacts'


def preflight():
    original = json.loads((BASELINE / 'config.json').read_text(encoding='utf-8'))
    for rel in ('word_energy/fresh_synthetic.py', 'word_energy/ising/spectral_initializer.py',
                'word_energy/ising/exact_training.py', 'scripts/diagnose_ising_validation.py'):
        path = ROOT / rel
        if sha256_file(path) != sha256_file(BASELINE / 'source_snapshot' / path.name):
            raise ValueError(f'baseline source mismatch: {rel}')
    data = prepare_data(corpus=make_fresh_corpus(227))
    spectral = spectral_biases(data['train_windows'], len(data['ids']))
    if (data['protocol_sha256'] != original['protocol_sha256'] or
        data['schedule_sha256'] != original['schedule_sha256'] or
        array_hash(spectral['biases']) != original['initializer']['bias_hash']):
        raise ValueError('baseline data, schedule or spectral geometry differs')
    prior = MlflowClient().get_experiment_by_name('word-energy-fresh-synthetic-v2')
    if prior is None:
        raise ValueError('baseline MLflow experiment missing')
    original_runs = MlflowClient().search_runs([prior.experiment_id], max_results=1000)
    for seed in SEEDS:
        matches = [r for r in original_runs if r.data.tags.get('replicate') == '227'
                   and r.data.tags.get('seed') == str(seed) and r.data.tags.get('arm') == 'spectral']
        if len(matches) != 1 or matches[0].info.status != 'FINISHED':
            raise ValueError(f'baseline spectral run missing/duplicate for seed {seed}')
        config = json.loads((ROOT/'mlruns'/prior.experiment_id/matches[0].info.run_id/'artifacts'/'config.json').read_text(encoding='utf-8'))
        raw = random_initial(seed, len(data['ids']))
        if any(array_hash(raw[k]) != config['initial_hashes'][k] for k in ('internal', 'interaction')):
            raise ValueError(f'paired seed initialization mismatch: {seed}')
    indices = [tuple(np.random.default_rng(seed).permutation(len(data['ids'])).tolist()) for seed in PERMUTATIONS]
    if len(set(indices)) != len(PERMUTATIONS) or tuple(spectral['permutation']) in indices:
        raise ValueError('duplicate permutation or baseline permutation')
    return {'data_hash': data['protocol_sha256'], 'schedule_hash': data['schedule_sha256'],
            'bias_hash': array_hash(spectral['biases']), 'permutations': dict(zip(PERMUTATIONS, indices))}


if __name__ == '__main__':
    mlflow.set_tracking_uri(tracking_uri())
    checked = preflight()
    exp = mlflow.set_experiment('word-energy-corpus227-permutation-panel-v1')
    if MlflowClient().search_runs([exp.experiment_id], max_results=1000):
        raise RuntimeError('existing runs found; inspect before restarting')
    print('PREFLIGHT ' + json.dumps(checked), flush=True)
    env = os.environ.copy()
    env.update({'OMP_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4', 'MLFLOW_DISABLE_AGENT_HINT':'1'})
    start = time.monotonic()
    deadline = start + 2400
    total = 0
    for permutation_seed in PERMUTATIONS:
        for seed in SEEDS:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'panel incomplete: {total}/32')
            try:
                result = subprocess.run([sys.executable, str(ROOT/'scripts/run_ising_permutation_panel_arm.py'),
                                         str(seed), str(permutation_seed)], cwd=ROOT, env=env,
                                        text=True, capture_output=True, timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f'panel incomplete: {total}/32; interrupted run requires review') from exc
            if result.stdout:
                print(result.stdout.strip(), flush=True)
            if result.returncode:
                raise RuntimeError(f'failed permutation={permutation_seed}, seed={seed}: {result.stderr}')
            total += 1
    print(f'BATCH COMPLETE: runs={total}, wall_seconds={time.monotonic()-start:.2f}', flush=True)
