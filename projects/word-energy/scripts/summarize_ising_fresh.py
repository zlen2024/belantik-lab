"""Audit and summarize the frozen fresh-synthetic panel after it completes."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from word_energy.fresh_synthetic import REPLICATES
from word_energy.tracking import RunContract, log_contract, tracking_uri

SEEDS=(251,257,263,269); ARMS=('random','spectral','permuted')


def inspect():
    client=MlflowClient(); experiment=client.get_experiment_by_name('word-energy-fresh-synthetic-v2')
    if experiment is None: raise ValueError('benchmark experiment missing')
    runs=client.search_runs([experiment.experiment_id],max_results=1000)
    indexed={}
    for run in runs:
        tags=run.data.tags
        if not all(k in tags for k in ('replicate','seed','arm')): continue
        key=(int(tags['replicate']),int(tags['seed']),tags['arm'])
        if key in indexed: raise ValueError(f'duplicate arm: {key}')
        indexed[key]=run
    expected={(rep,seed,arm) for rep in REPLICATES for seed in SEEDS for arm in ARMS}
    if set(indexed)!=expected: raise ValueError(f'incomplete or unexpected runs: missing={sorted(expected-set(indexed))}, unexpected={sorted(set(indexed)-expected)}')
    output={}; paired_data={}
    for (rep,seed,arm),run in sorted(indexed.items()):
        key=f'{rep}:{seed}:{arm}'
        if run.info.status=='FAILED' and run.data.tags.get('status_reason')=='spectral_eigengap_failure' and arm!='random':
            output[key]={'run_id':run.info.run_id,'status':'predeclared_eigengap_failure','joint100_pass':False}
            continue
        if run.info.status!='FINISHED': raise ValueError(f'{key}: status {run.info.status}')
        path=ROOT/'mlruns'/experiment.experiment_id/run.info.run_id/'artifacts'/'config.json'
        payload=json.loads(path.read_text(encoding='utf-8'))
        if (payload['replicate'],payload['seed'],payload['arm'])!=(rep,seed,arm): raise ValueError(f'{key}: config identity mismatch')
        if payload['lexical_config']['normalize_spokes'] is not True or payload['joint_config']['normalize_spokes'] is not True: raise ValueError(f'{key}: wrong F3')
        if payload['lexical_config']['lexical_updates']!=400 or payload['joint_config']['joint_updates']!=100: raise ValueError(f'{key}: wrong update budget')
        if payload['snapshots']['stage1_lexical_400']!=payload['snapshots']['stage2_initial_0']: raise ValueError(f'{key}: handoff mismatch')
        if payload['joint_metrics']['protocol.sha256']!=payload['protocol_sha256']: raise ValueError(f'{key}: train data hash mismatch')
        m=payload['fixed_joint100']; lex=payload['fixed_lexical400']
        computed=bool(m['sense']>=10/12 and m['auc']>=.75 and m['gain']>=.03)
        if computed!=m['pass']: raise ValueError(f'{key}: gate mismatch')
        output[key]={'run_id':run.info.run_id,'status':run.info.status,'lexical400':lex,'joint100':m,'joint100_pass':computed,'peak_process_working_set_bytes':payload['peak_process_working_set_bytes'],'elapsed_seconds':payload['elapsed_seconds']}
        group=paired_data.setdefault((rep,seed),{})
        group[arm]={'protocol':payload['protocol_sha256'],'train_hash':payload['train_token_hash'],'schedule':payload['schedule_sha256'],'internal':payload['initial_hashes']['internal'],'interaction':payload['initial_hashes']['interaction'],'initializer':payload['initializer']}
    for key,arms in paired_data.items():
        for field in ('protocol','train_hash','schedule','internal','interaction'):
            if len({v[field] for v in arms.values()})!=1: raise ValueError(f'{key}: paired {field} mismatch')
        if 'spectral' in arms and 'permuted' in arms:
            a,b=arms['spectral']['initializer'],arms['permuted']['initializer']
            for field in ('count_hash','eigenvectors_hash','bias_hash','permuted_bias_hash','permutation'):
                if a[field]!=b[field]: raise ValueError(f'{key}: spectral control {field} mismatch')
    rates={str(rep):{arm:float(np.mean([output[f'{rep}:{seed}:{arm}']['joint100_pass'] for seed in SEEDS])) for arm in ARMS} for rep in REPLICATES}
    differences={str(rep):{'spectral_minus_random':rates[str(rep)]['spectral']-rates[str(rep)]['random'],'spectral_minus_permuted':rates[str(rep)]['spectral']-rates[str(rep)]['permuted']} for rep in REPLICATES}
    promotion=all(d['spectral_minus_random']>0 and d['spectral_minus_permuted']>0 for d in differences.values())
    return {'panel_size':len(output),'runs':output,'per_corpus_pass_rates':rates,'per_corpus_differences':differences,'strict_six_corpus_promotion':promotion,'sign_test_p_each_if_promoted':1/64 if promotion else None,'bonferroni_p_if_promoted':2/64 if promotion else None,'limitations':'n=6 synthetic corpus replicates, 4 seeds per corpus; fixed row permutation; same oracle-assisted grammar, tokens seen in train, no test or hardware'}


if __name__=='__main__':
    mlflow.set_tracking_uri(tracking_uri()); summary=inspect()
    mlflow.set_experiment('word-energy-fresh-synthetic-v2')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='ising_fresh_synthetic_v2_panel_summary',energy_convention='beta E; corrected F3 lambda/2',variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',sampler_schedule='read-only aggregation of completed panel',seed=0,data_hash='six individual protocol hashes in source runs',data_split='fresh synthetic validation only',graph_summary='3 token star, 2 spins each',limitations=summary['limitations'])
    with mlflow.start_run(run_name='fresh-v2-panel-summary') as run:
        log_contract(mlflow,contract,summary)
        mlflow.log_metric('strict_six_corpus_promotion',float(summary['strict_six_corpus_promotion']))
        for rep,arms in summary['per_corpus_pass_rates'].items(): mlflow.log_metrics({f'corpus{rep}.{arm}.pass_rate':rate for arm,rate in arms.items()})
        print(json.dumps({'summary_run_id':run.info.run_id,'panel_size':summary['panel_size'],'pass_rates':summary['per_corpus_pass_rates'],'differences':summary['per_corpus_differences'],'strict_promotion':summary['strict_six_corpus_promotion']}))
