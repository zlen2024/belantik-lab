"""Record saved attribution trajectories and validate fixed endpoints."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mlflow
from scripts.diagnose_ising_validation import diagnose_raw
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri

if __name__ == '__main__':
    seed = int(sys.argv[1])
    branch = sys.argv[2]
    artifact = ROOT / 'experiments' / 'runs' / f'ising-attribution-{branch}-seed{seed}.json'
    payload = json.loads(artifact.read_text(encoding='utf-8'))
    replay = json.loads((ROOT / 'experiments' / 'runs' / f'ising-attribution-original-seed{seed}.json').read_text(encoding='utf-8'))
    if not replay['reproduction_pass']:
        raise ValueError('cannot report attribution without successful replay')
    labels = ['joint_100']
    if branch == 'original':
        labels = ['initial_0', f"lexical_selected_{payload['selected_metrics']['selection.lexical_step']}", 'joint_100', f"joint_selected_{payload['selected_metrics']['selection.joint_step']}"]
    results = {}
    for label in dict.fromkeys(labels):
        raw = {key: np.asarray(value, dtype=np.float64) for key, value in payload['snapshots'][label].items()}
        results[label] = diagnose_raw(raw, seed, f'{branch}:{label}', sha256_file(artifact))
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment('word-energy-exact-training')
    contract = RunContract(execution_tier='classical', device_class='cpu_gpu_simulator', model_family='ising_exact_attribution', energy_convention='dimensionless beta E; exact Ising ranking', variable_encoding='spins {-1,+1}', beta_temperature_convention='beta=1', sampler_schedule='exact enumeration; deterministic training replay', seed=seed, data_hash=sha256_file(ROOT / 'experiments' / 'runs' / 'synthetic-split-manifest.json'), data_split='training and validation only; train anchors for region mapping', graph_summary='3 tokens, 2 spins per token, star graph', limitations='Exploratory validation used for selection; synthetic data, no hardware claim; intermediate snapshots saved but endpoint diagnostics only.')
    with mlflow.start_run(run_name=f'ising-attribution-{branch}-seed{seed}') as run:
        log_contract(mlflow, contract, {'branch': branch, 'checkpoint_sha256': payload['original_checkpoint_sha256'], 'replay_parameter_difference': replay['selected_parameter_max_abs_difference_from_original'], 'diagnostics': results, 'selected_metrics': payload['selected_metrics']})
        metrics = {}
        for label, diag in results.items():
            for key in ('anchor_total_variation','context_posterior_total_variation','validation_macro_sense_accuracy','validation_joint_ranking_loss'):
                metrics[f'{label}.{key}'] = diag[key]
            for family, values in diag['family_summary'].items():
                for key, value in values.items():
                    metrics[f'{label}.{family}.{key}'] = value
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(artifact))
        print(json.dumps({'run_id':run.info.run_id, 'branch':branch, 'seed':seed, 'endpoints': {label: {'anchor_tv':d['anchor_total_variation'], 'context_tv':d['context_posterior_total_variation'], 'accuracy':d['validation_macro_sense_accuracy']} for label,d in results.items()}}))
