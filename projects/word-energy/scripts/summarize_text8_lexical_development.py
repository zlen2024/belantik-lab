"""Audit frozen samples/checkpoints and summarize the completed text8 pilot."""
from pathlib import Path
import json
import sys
import numpy as np
from urllib.parse import urlparse
from urllib.request import url2pathname

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.train_text8_lexical_development import get_data
from word_energy.text8_development import sha_array
from word_energy.text8_lexical_models import ARMS,SEEDS
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file


def artifact_dir(run):
    uri=urlparse(run.info.artifact_uri)
    if uri.scheme!='file': raise ValueError('expected local artifact store')
    return Path(url2pathname(uri.path))


def inspect(data_run):
    client=MlflowClient(); arrays,manifest=get_data(data_run)
    exp=client.get_experiment_by_name('word-energy-text8-lexical-development')
    if exp is None: raise ValueError('experiment missing')
    indexed={}
    for run in client.search_runs([exp.experiment_id],max_results=1000):
        t=run.data.tags
        if not all(k in t for k in ('arm','seed')): continue
        key=(t['arm'],int(t['seed']))
        if key in indexed: raise ValueError(f'duplicate arm/seed {key}')
        indexed[key]=run
    expected={(arm,seed) for arm in ARMS for seed in SEEDS}
    if set(indexed)!=expected: raise ValueError(f'incomplete/unexpected: missing={sorted(expected-set(indexed))}; extra={sorted(set(indexed)-expected)}')
    rows={}; source_hashes={}; initial_hashes={}
    for (arm,seed),run in sorted(indexed.items()):
        if run.info.status!='FINISHED': raise ValueError(f'failed or unfinished {arm}/{seed}')
        root=artifact_dir(run); cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
        if not cfg['run_complete'] or cfg['arm']!=arm or cfg['seed']!=seed or cfg['dataset_run_id']!=data_run:
            raise ValueError('run identity or completion mismatch')
        if cfg['dataset_hashes']!=manifest['hashes'] or cfg['epochs']!=3 or cfg['batch']!=256 or cfg['checkpoint_rule']!='fixed epoch3':
            raise ValueError('data/budget mismatch')
        expected_params=(3*len(manifest['vocabulary'])+2) if arm.startswith('ising') else 4*len(manifest['vocabulary'])
        if cfg['parameter_count']!=expected_params: raise ValueError('parameter count mismatch')
        for stage in ('initial','epoch1','epoch2','epoch3'):
            with np.load(root/'checkpoints'/f'{stage}.npz',allow_pickle=False) as checkpoint:
                actual={key:sha_array(checkpoint[key]) for key in checkpoint.files}
            if actual!=cfg['checkpoint_hashes'][stage]: raise ValueError(f'checkpoint hash mismatch {arm}/{seed}/{stage}')
            if abs(run.data.metrics[f'{stage}.noise_pairwise_auc']-cfg['metrics'][stage]['noise_pairwise_auc'])>1e-12:
                raise ValueError('metric/database mismatch')
        initial_hashes[(arm,seed)]=cfg['checkpoint_hashes']['initial']
        for path in (root/'source_snapshot').iterdir():
            source_hashes.setdefault(path.name,set()).add(sha256_file(path))
        rows[f'{arm}:{seed}']={'run_id':run.info.run_id,'parameter_count':cfg['parameter_count'],
                               'metrics':cfg['metrics'],'elapsed_seconds':cfg['elapsed_seconds'],
                               'checkpoint_hashes':cfg['checkpoint_hashes']}
    if any(len(values)!=1 for values in source_hashes.values()): raise ValueError('model source versions differ')
    for seed in SEEDS:
        if initial_hashes[('ising-ranking',seed)]!=initial_hashes[('ising-shuffled',seed)]: raise ValueError('Ising/shuffled initialization differs')
        if initial_hashes[('vector-ranking',seed)]!=initial_hashes[('sgns',seed)]: raise ValueError('vector initialization differs')
    aggregate={}
    for arm in ARMS:
        values=np.asarray([rows[f'{arm}:{seed}']['metrics']['epoch3']['noise_pairwise_auc'] for seed in SEEDS])
        mrr=np.asarray([rows[f'{arm}:{seed}']['metrics']['epoch3']['noise_mrr_at_21'] for seed in SEEDS])
        aggregate[arm]={'noise_auc_mean':float(values.mean()),'noise_auc_seed_sd':float(values.std(ddof=1)),
                        'noise_mrr_mean':float(mrr.mean()),'noise_mrr_seed_sd':float(mrr.std(ddof=1)),
                        'seed_auc':values.tolist(),'seed_mrr':mrr.tolist()}
    return {'dataset_run_id':data_run,'dataset_hashes':manifest['hashes'],'source_sha256':manifest['source']['content_sha256'],
            'train_stats':manifest['train_stats'],'validation_stats':manifest['validation_stats'],
            'train_unique_pairs':manifest['train_unique_pairs'],'validation_unique_pairs':manifest['validation_unique_pairs'],
            'train_unique_occurrence_pairs':manifest['train_unique_occurrence_pairs'],'validation_unique_occurrence_pairs':manifest['validation_unique_occurrence_pairs'],
            'validation_exact_noise_collisions':manifest['validation_exact_noise_collisions'],
            'frequency_only_control':manifest['frequency_only_control'],'runs':rows,'aggregate':aggregate,
            'limitations':'text8 development only, no verified document independence, shared corpus across seeds; score is validation noise discrimination, not sense or hardware; parameter budgets and SGNS objective differ.'}


if __name__=='__main__':
    data_run=sys.argv[1]; mlflow.set_tracking_uri(tracking_uri()); result=inspect(data_run)
    mlflow.set_experiment('word-energy-text8-lexical-development')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='text8_lexical_panel_summary',
        energy_convention='Ising beta E or vector dot score',variable_encoding='spins {-1,+1} or 2D vectors',
        beta_temperature_convention='beta=1 for Ising',sampler_schedule='fixed noise pairs and exact Ising F2 training',seed=0,
        data_hash=result['source_sha256'],data_split='text8 development validation only, reserved test not evaluated',
        graph_summary='2-token 2-spin Ising F2 or two 2D vector tables',limitations=result['limitations'])
    with mlflow.start_run(run_name='text8-lexical-development-summary') as run:
        log_contract(mlflow,contract,result)
        mlflow.log_artifact(str(Path(__file__).resolve()),'source_snapshot')
        mlflow.log_metrics({f"{arm}.noise_auc_mean":v['noise_auc_mean'] for arm,v in result['aggregate'].items()})
        print(json.dumps({'summary_run_id':run.info.run_id,'frequency_control':result['frequency_only_control']['noise_pairwise_auc'],'aggregate':result['aggregate']}))
