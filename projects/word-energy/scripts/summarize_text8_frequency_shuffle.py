"""Independently summarize original vs shuffled development checkpoints."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.train_text8_lexical_development import get_data
from word_energy.text8_factorial_models import functions,SEEDS
from word_energy.text8_canonical_eval import canonical_candidate_scores
from word_energy.text8_development import sha_array
from word_energy.text8_lexical_models import pair_metrics
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file

DATA='6cf76591c3fa49c686a3d81d7f225607'
ARMS=('tied-frequency','role-frequency')

def audit():
    client=MlflowClient(); a,m=get_data(DATA); pairs=a['validation_pairs']; noise=a['validation_noise']; count=a['counts']; offset=.25*np.log(count.astype(float))
    original=client.get_experiment_by_name('word-energy-text8-frequency-role-factorial'); shuffle=client.get_experiment_by_name('word-energy-text8-frequency-shuffle-control')
    real=[r for r in client.search_runs([original.experiment_id],max_results=1000) if r.data.tags.get('arm') in ARMS]
    fake=[r for r in client.search_runs([shuffle.experiment_id],max_results=1000) if r.data.tags.get('arm') in ARMS]
    if len(real)!=6 or len(fake)!=6: raise ValueError('expected 6 original and 6 shuffled')
    diff_ids=pairs[:,1,None]!=noise
    ratio=count[pairs[:,1,None]]/count[noise]
    groups={'matched_0_5_to_2':diff_ids & (ratio>=.5) & (ratio<=2.),'equal_count_distinct_ids':diff_ids & (ratio==1.)}
    sizes={k:int(np.count_nonzero(v)) for k,v in groups.items()}; results={}; inits={}; sources={}
    for group,runs in [('original',real),('shuffled',fake)]:
        for run in runs:
            arm=run.data.tags['arm']; seed=int(run.data.tags['seed']); key=f'{group}:{arm}:{seed}'
            if key in results or run.info.status!='FINISHED': raise ValueError('unfinished/duplicate '+key)
            root=artifact_dir(client,run.info.run_id); config=json.loads((root/'config.json').read_text(encoding='utf-8'))
            if not config['run_complete'] or config['dataset_hashes']!=m['hashes'] or config['offset_sha256']!=sha_array(offset): raise ValueError('dataset/config mismatch')
            if group=='shuffled' and config['training_pairs']!='shuffled_train_pairs': raise ValueError('wrong data')
            if config['parameter_count']!=(6002 if arm.startswith('role') else 3002): raise ValueError('parameter mismatch')
            source={p.name:sha256_file(p) for p in (root/'source_snapshot').iterdir() if p.is_file()}
            if group in sources and sources[group]!=source: raise ValueError('sources inconsistent')
            sources[group]=source
            loaded={}
            for stage in ('initial','epoch1','epoch2','epoch3'):
                with np.load(root/'checkpoints'/f'{stage}.npz',allow_pickle=False) as z: p={k:z[k] for k in z.files}
                if {k:sha_array(v) for k,v in p.items()}!=config['checkpoint_hashes'][stage]: raise ValueError('checkpoint mismatch')
                if stage in ('initial','epoch3'): loaded[stage]=p
            inits[key]=loaded['initial']; score=functions(arm,offset)[0]
            pos,neg=canonical_candidate_scores(loaded['epoch3'],score,pairs,noise,symmetric=False)
            metric=pair_metrics(pos,neg,count[pairs[:,0]])
            for k in ('noise_pairwise_auc','noise_mrr_at_21'):
                if not np.isclose(metric[k],config['metrics']['epoch3'][k],atol=1e-14,rtol=0): raise ValueError('metric mismatch')
            win=(pos[:,None]>neg).astype(float)+.5*(pos[:,None]==neg)
            subgroup={k:float(win[mask].mean()) for k,mask in groups.items()}
            results[key]={'run_id':run.info.run_id,'metrics':metric,'subgroup_auc':subgroup,'elapsed_seconds':config['elapsed_seconds'],'peak_mib':config['metrics']['epoch3']['peak_working_set_mib']}
    effects={}
    for arm in ARMS:
        effects[arm]={}
        for key in ('noise_pairwise_auc','noise_mrr_at_21','matched_0_5_to_2','equal_count_distinct_ids'):
            values=[]
            for seed in SEEDS:
                r=results[f'original:{arm}:{seed}']; q=results[f'shuffled:{arm}:{seed}']; oi=inits[f'original:{arm}:{seed}']; si=inits[f'shuffled:{arm}:{seed}']
                if set(oi)!=set(si) or any(not np.array_equal(oi[k],si[k]) for k in oi): raise ValueError('initial mismatch')
                source='metrics' if key.startswith('noise_') else 'subgroup_auc'
                values.append(r[source][key]-q[source][key])
            effects[arm][key]={'per_seed':values,'mean':float(np.mean(values)),'auc_primary_material':bool(all(v>0 for v in values) and np.mean(values)>=.005) if key=='noise_pairwise_auc' else None}
    return {'dataset_run_id':DATA,'source_sha256':m['source']['content_sha256'],'dataset_hashes':m['hashes'],'source_hashes_by_group':sources,'subgroup_comparison_counts':sizes,'results':results,'effects':effects,'limitations':'posthoc development; one fixed shuffle; common validation; comparison-level groups correlated; no semantic/hardware claim'}

if __name__=='__main__':
    start=time.monotonic(); mlflow.set_tracking_uri(tracking_uri()); result=audit(); result['audit_seconds']=time.monotonic()-start
    mlflow.set_experiment('word-energy-text8-frequency-shuffle-control')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='text8_frequency_shuffle_summary',energy_convention='normalized F2 plus candidate frequency offset',variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',sampler_schedule='audit existing exact checkpoints',seed=0,data_hash=result['source_sha256'],data_split='existing development validation; reserved test untouched',graph_summary='four-spin F2',limitations=result['limitations'])
    with mlflow.start_run(run_name='text8-frequency-shuffle-summary') as run:
        log_contract(mlflow,contract,result)
        for arm,metrics in result['effects'].items():
            for name,detail in metrics.items(): mlflow.log_metric(f'{arm}.{name}.paired_mean',detail['mean'])
        mlflow.log_artifact(str(Path(__file__).resolve()),'source_snapshot')
        print(json.dumps({'run_id':run.info.run_id,'subgroup_comparison_counts':result['subgroup_comparison_counts'],'effects':result['effects'],'results':result['results']},indent=2))
