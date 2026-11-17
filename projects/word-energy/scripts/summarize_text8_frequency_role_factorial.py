"""Verify factorial provenance and record paired development effects in MLflow."""
from pathlib import Path
import json, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.train_text8_lexical_development import get_data
from word_energy.text8_development import sha_array
from word_energy.text8_factorial_models import ARMS,SEEDS
from word_energy.tracking import RunContract,log_contract,sha256_file,tracking_uri

DATA='6cf76591c3fa49c686a3d81d7f225607'
def audit():
    client=MlflowClient(); arrays,manifest=get_data(DATA)
    exp=client.get_experiment_by_name('word-energy-text8-frequency-role-factorial')
    runs=client.search_runs([exp.experiment_id],max_results=1000)
    runs=[r for r in runs if r.data.tags.get('arm') in ARMS]
    if len(runs)!=12: raise ValueError('expected precisely 12 runs')
    results={}; checkpoints={}; baseline_sources=None
    for run in runs:
        arm=run.data.tags['arm']; seed=int(run.data.tags['seed']); key=f'{arm}:{seed}'
        if key in results or run.info.status!='FINISHED': raise ValueError('duplicate or unfinished '+key)
        root=artifact_dir(client,run.info.run_id); config=json.loads((root/'config.json').read_text(encoding='utf-8'))
        if not config['run_complete'] or config['dataset_hashes']!=manifest['hashes'] or config['dataset_run_id']!=DATA: raise ValueError('run/data mismatch')
        if config['parameter_count']!=(6002 if arm.startswith('role') else 3002): raise ValueError('parameter count')
        sources={p.name:sha256_file(p) for p in (root/'source_snapshot').iterdir() if p.is_file()}
        if baseline_sources is None: baseline_sources=sources
        if sources!=baseline_sources: raise ValueError('source snapshots changed across runs')
        for stage in ('initial','epoch1','epoch2','epoch3'):
            with np.load(root/'checkpoints'/f'{stage}.npz',allow_pickle=False) as a: params={k:a[k] for k in a.files}
            if {k:sha_array(v) for k,v in params.items()}!=config['checkpoint_hashes'][stage]: raise ValueError('checkpoint hashes')
            for metric in ('noise_pairwise_auc','noise_mrr_at_21'):
                if config['metrics'][stage][metric]!=run.data.metrics[f'{stage}.{metric}']: raise ValueError('metric mismatch')
            if stage=='initial': checkpoints[key]=params
        zero=config['zero_coupling_control']; expected=manifest['frequency_only_control'] if arm.endswith('frequency') else {'noise_pairwise_auc':.5,'noise_mrr_at_21':sum(1/i for i in range(1,22))/21}
        if any(not np.isclose(zero[k],expected[k],atol=1e-14,rtol=0) for k in ('noise_pairwise_auc','noise_mrr_at_21')): raise ValueError('zero control')
        results[key]={'run_id':run.info.run_id,'config':config}
    for seed in SEEDS:
        tied=checkpoints[f'tied:{seed}']; role=checkpoints[f'role:{seed}']
        for a,b in [('tied','tied-frequency'),('role','role-frequency')]:
            if any(not np.array_equal(v,checkpoints[f'{b}:{seed}'][k]) for k,v in checkpoints[f'{a}:{seed}'].items()): raise ValueError('offset pair initial mismatch')
        for old,new in [('biases','target_biases'),('biases','context_biases'),('internal','target_internal'),('internal','context_internal'),('interaction','interaction')]:
            if not np.array_equal(tied[old],role[new]): raise ValueError('role init mismatch')
    effects={}
    for metric in ('noise_pairwise_auc','noise_mrr_at_21'):
        values={}
        for seed in SEEDS:
            a,b,c,d=[results[f'{arm}:{seed}']['config']['metrics']['epoch3'][metric] for arm in ARMS]
            values[str(seed)]={'frequency_when_tied':b-a,'frequency_when_role':d-c,'role_when_no_frequency':c-a,'role_when_frequency':d-b,'interaction':d-c-b+a}
        effects[metric]={}
        for name in next(iter(values.values())):
            v=[values[str(seed)][name] for seed in SEEDS]; avg=float(np.mean(v))
            effects[metric][name]={'per_seed':v,'mean':avg,'all_positive_and_mean_at_least_0_005_auc_gate':bool(all(x>0 for x in v) and avg>=.005) if metric=='noise_pairwise_auc' else None}
    return {'dataset_run_id':DATA,'source_sha256':manifest['source']['content_sha256'],'dataset_hashes':manifest['hashes'],
            'source_snapshot_sha256':baseline_sources,'runs':results,'effects':effects,
            'limitations':'posthoc development, correlated seeds; role capacity differs; host frequency offset; no sense or test or hardware claim'}

if __name__=='__main__':
    start=time.monotonic(); mlflow.set_tracking_uri(tracking_uri()); out=audit(); out['audit_seconds']=time.monotonic()-start
    mlflow.set_experiment('word-energy-text8-frequency-role-factorial')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='text8_factorial_summary',energy_convention='normalized F2 and optional candidate offset',variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',sampler_schedule='audit existing exact checkpoints',seed=0,data_hash=out['source_sha256'],data_split='existing development validation; reserved test untouched',graph_summary='four-spin F2',limitations=out['limitations'])
    with mlflow.start_run(run_name='text8-factorial-paired-summary') as run:
        log_contract(mlflow,contract,out); mlflow.log_artifact(str(Path(__file__).resolve()),'source_snapshot')
        for metric,items in out['effects'].items():
            for name,v in items.items(): mlflow.log_metric(f'{metric}.{name}.mean',v['mean'])
        print(json.dumps({'run_id':run.info.run_id,'effects':out['effects'],'runs':{k:{'id':v['run_id'],'auc':v['config']['metrics']['epoch3']['noise_pairwise_auc'],'mrr':v['config']['metrics']['epoch3']['noise_mrr_at_21'],'seconds':v['config']['elapsed_seconds']} for k,v in out['runs'].items()}},indent=2))
