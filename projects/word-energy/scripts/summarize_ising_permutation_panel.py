"""Audit and summarize the fixed exploratory corpus-227 permutation panel."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.run_ising_permutation_panel_batch import PERMUTATIONS, preflight
from scripts.run_ising_fresh_synthetic import SEEDS
from word_energy.ising.spectral_initializer import array_hash
from word_energy.tracking import RunContract, log_contract, tracking_uri


def inspect():
    client = MlflowClient()
    exp = client.get_experiment_by_name('word-energy-corpus227-permutation-panel-v1')
    baseline_exp = client.get_experiment_by_name('word-energy-fresh-synthetic-v2')
    if exp is None or baseline_exp is None:
        raise ValueError('experiment missing')
    indexed = {}
    for run in client.search_runs([exp.experiment_id], max_results=1000):
        tags = run.data.tags
        if 'permutation_seed' not in tags or 'seed' not in tags:
            continue
        key = (int(tags['permutation_seed']), int(tags['seed']))
        if key in indexed:
            raise ValueError(f'duplicate run {key}')
        indexed[key] = run
    expected = {(p, s) for p in PERMUTATIONS for s in SEEDS}
    if set(indexed) != expected:
        raise ValueError(f'incomplete panel: missing={sorted(expected-set(indexed))}; extra={sorted(set(indexed)-expected)}')
    verified = preflight()
    prior = {}
    for run in client.search_runs([baseline_exp.experiment_id], max_results=1000):
        tags = run.data.tags
        if tags.get('replicate') == '227' and tags.get('arm') in ('random','spectral','permuted') and tags.get('seed') in map(str, SEEDS):
            prior[(tags['arm'],int(tags['seed']))] = run
    if len(prior) != 12:
        raise ValueError('missing baseline runs')
    old_configs = {}
    for key, run in prior.items():
        if run.info.status != 'FINISHED':
            raise ValueError('unfinished baseline')
        path = ROOT/'mlruns'/baseline_exp.experiment_id/run.info.run_id/'artifacts'/'config.json'
        old_configs[key] = json.loads(path.read_text(encoding='utf-8'))
    result = {}
    for (p, seed), run in sorted(indexed.items()):
        if run.info.status != 'FINISHED':
            raise ValueError(f'{p}:{seed} is {run.info.status}')
        path = ROOT/'mlruns'/exp.experiment_id/run.info.run_id/'artifacts'/'config.json'
        config = json.loads(path.read_text(encoding='utf-8'))
        old = old_configs[('spectral',seed)]
        if (config['replicate'],config['seed'],config['permutation_seed']) != (227,seed,p):
            raise ValueError('identity mismatch')
        for field in ('protocol_sha256','schedule_sha256','train_token_hash','lexical_config','joint_config'):
            if config[field] != old[field]:
                raise ValueError(f'{p}:{seed}: baseline mismatch {field}')
        for field in ('internal','interaction'):
            if config['initial_hashes'][field] != old['initial_hashes'][field]:
                raise ValueError(f'{p}:{seed}: paired initial {field}')
        expected_perm = list(verified['permutations'][p])
        if config['initializer']['permutation'] != expected_perm:
            raise ValueError(f'{p}:{seed}: wrong permutation')
        initial = np.asarray(config['snapshots']['stage1_initial_0']['biases'],dtype=np.float64)
        old_initial = np.asarray(old['snapshots']['stage1_initial_0']['biases'],dtype=np.float64)
        if not np.array_equal(initial, old_initial[expected_perm]) or array_hash(initial) != config['initial_hashes']['biases']:
            raise ValueError(f'{p}:{seed}: actual bias permutation mismatch')
        if config['snapshots']['stage1_lexical_400'] != config['snapshots']['stage2_initial_0']:
            raise ValueError(f'{p}:{seed}: handoff mismatch')
        for stage in ('fixed_lexical400','fixed_joint100'):
            m=config[stage]; passed=bool(m['sense'] >= 10/12 and m['auc'] >= .75 and m['gain'] >= .03)
            if m['pass'] != passed:
                raise ValueError(f'{p}:{seed}: incorrect pass')
        result[f'{p}:{seed}']={'run_id':run.info.run_id, 'lexical400':config['fixed_lexical400'],
                               'joint100':config['fixed_joint100'], 'peak_bytes':config['peak_process_working_set_bytes'],
                               'elapsed_seconds':config['elapsed_seconds']}
    old_result = {f'{arm}:{seed}': {'run_id':prior[(arm,seed)].info.run_id,
        'lexical400':old_configs[(arm,seed)]['fixed_lexical400'],
        'joint100':old_configs[(arm,seed)]['fixed_joint100']} for arm,seed in sorted(prior)}
    rates = {str(p):sum(result[f'{p}:{seed}']['joint100']['pass'] for seed in SEEDS) for p in PERMUTATIONS}
    return {'runs':result,'permutation_pass_counts':rates,'previous_controls':old_result,
            'limitations':'exploratory posthoc single selected corpus; 8 permutations, 4 paired model seeds, oracle-assisted toy, no test or hardware'}


if __name__ == '__main__':
    mlflow.set_tracking_uri(tracking_uri())
    summary=inspect()
    mlflow.set_experiment('word-energy-corpus227-permutation-panel-v1')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='ising_corpus227_permutation_panel_summary',energy_convention='beta E; corrected F3 lambda/2',variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',sampler_schedule='aggregation only; exact enumeration source runs',seed=0,data_hash='see individual runs',data_split='selected corpus 227 validation only',graph_summary='three token star, two spins each',limitations=summary['limitations'])
    with mlflow.start_run(run_name='corpus227-permutation-panel-summary') as run:
        log_contract(mlflow,contract,summary)
        mlflow.log_metrics({f'perm{p}.pass_count':float(n) for p,n in summary['permutation_pass_counts'].items()})
        print(json.dumps({'summary_run_id':run.info.run_id, 'permutation_pass_counts':summary['permutation_pass_counts']}))
