"""Reproduce the frozen Ising run before executing validation-only ablations."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from word_energy.ising.exact_training import ExactTrainingConfig, train_smoke

if __name__ == '__main__':
    seed = int(sys.argv[1])
    branch = sys.argv[2]
    if branch not in ('original', 'joint_from_initial', 'lexical_auxiliary'):
        raise ValueError('unknown branch')
    snapshots = {}
    def capture(stage, step, raw):
        if not all(np.all(np.isfinite(value)) for value in raw.values()):
            raise FloatingPointError(f'nonfinite parameters at {stage} step {step}')
        snapshots[f'{stage}_{step}'] = raw
    start = time.monotonic()
    override = None
    config = ExactTrainingConfig(seed=seed)
    if branch != 'original':
        replay_path = ROOT / 'experiments' / 'runs' / f'ising-attribution-original-seed{seed}.json'
        replay = json.loads(replay_path.read_text(encoding='utf-8'))
        if not replay['reproduction_pass']:
            raise ValueError('original replay must pass before running branches')
        source = 'initial_0' if branch == 'joint_from_initial' else f"lexical_selected_{replay['selected_metrics']['selection.lexical_step']}"
        override = {key: np.asarray(value, dtype=np.float64) for key, value in replay['snapshots'][source].items()}
        config = ExactTrainingConfig(seed=seed, lexical_updates=0)
    selected, metrics = train_smoke(config, snapshot_callback=capture, branch=branch, initial_override=override)
    metrics = {key: (None if isinstance(value, float) and not np.isfinite(value) else value) for key, value in metrics.items()}
    checkpoint = ROOT / 'experiments' / 'runs' / f'ising-exact-training-seed{seed}.npz'
    with np.load(checkpoint) as saved:
        differences = {key: float(np.max(np.abs(selected[key] - saved[key]))) for key in selected}
    max_difference = max(differences.values())
    reference = json.loads(checkpoint.with_suffix('.json').read_text(encoding='utf-8'))['metrics']
    keys = ('selection.lexical_step', 'selection.joint_step', 'validation.macro_sense_accuracy', 'validation.joint_ranking_loss')
    metric_differences = {key: abs(float(metrics[key]) - float(reference[key])) for key in keys} if branch == 'original' else {}
    result = {'seed': seed, 'branch': branch, 'elapsed_seconds': time.monotonic()-start, 'config': config.__dict__, 'selected_parameter_max_abs_difference_from_original': max_difference, 'replay_metric_abs_differences': metric_differences, 'reproduction_pass': max_difference <= 1e-10 and max(metric_differences.values()) <= 1e-10 if branch == 'original' else None, 'selected_metrics': metrics, 'original_checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'snapshots': {label: {key: value.tolist() for key, value in raw.items()} for label, raw in snapshots.items()}}
    destination = ROOT / 'experiments' / 'runs' / f'ising-attribution-{branch}-seed{seed}.json'
    if destination.exists():
        raise FileExistsError(f'refusing to overwrite {destination}')
    destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'snapshots'}, indent=2))
