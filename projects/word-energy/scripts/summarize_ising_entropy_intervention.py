"""Audit all frozen entropy-intervention runs before summarizing outcomes."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.run_ising_entropy_intervention_batch import PERMUTATIONS, ARMS, old_runs
from scripts.run_ising_fresh_synthetic import SEEDS
from word_energy.tracking import RunContract, log_contract, tracking_uri


def inspect():
    client=MlflowClient(); exp=client.get_experiment_by_name('word-energy-entropy-intervention-v1')
    if exp is None: raise ValueError('experiment missing')
    attempts={}
    for run in client.search_runs([exp.experiment_id],max_results=1000):
        tags=run.data.tags
        if not all(k in tags for k in ('permutation_seed','seed','arm')): continue
        key=(int(tags['permutation_seed']),int(tags['seed']),tags['arm'])
        attempts.setdefault(key,[]).append(run)
    indexed={}
    interrupted={}
    for key,runs in attempts.items():
        finished=[run for run in runs if run.info.status=='FINISHED']
        other=[run for run in runs if run.info.status!='FINISHED']
        if len(finished)!=1: raise ValueError(f'expected one finished attempt for {key}')
        if other:
            if len(other)!=1 or other[0].info.status!='KILLED' or finished[0].data.tags.get('retry_of_run_id')!=other[0].info.run_id:
                raise ValueError(f'unlinked or failed retry for {key}')
            interrupted[str(key)]=other[0].info.run_id
        indexed[key]=finished[0]
    expected={(p,s,a) for p in PERMUTATIONS for s in SEEDS for a in ARMS}
    if set(indexed)!=expected: raise ValueError(f'incomplete/unexpected panel: missing={sorted(expected-set(indexed))}; extra={sorted(set(indexed)-expected)}')
    old=old_runs(client); results={}
    for (perm,seed,arm),run in sorted(indexed.items()):
        if run.info.status!='FINISHED': raise ValueError(f'incomplete {perm}:{seed}:{arm}')
        cfg=json.loads((ROOT/'mlruns'/exp.experiment_id/run.info.run_id/'artifacts'/'config.json').read_text(encoding='utf-8'))
        prior=old[(perm,seed)]
        if (cfg['permutation_seed'],cfg['seed'],cfg['arm'])!=(perm,seed,arm): raise ValueError('identity mismatch')
        for field in ('protocol_sha256','schedule_sha256','train_token_hash','initial_hashes','lexical_config','joint_config'):
            if cfg[field]!=prior[field]: raise ValueError(f'{perm}:{seed}:{arm} mismatch {field}')
        expected_intervention={'target_id':cfg['vocabulary'].index('song' if arm=='early-song' else 'bank'),
                               'offset':100 if arm=='late-bank' else 0, 'strength':0.0 if arm=='baseline' else 0.1}
        if cfg['intervention']!=expected_intervention: raise ValueError('intervention mismatch')
        snaps=cfg['snapshots']
        if snaps['stage1_lexical_400']!=snaps['stage2_initial_0']: raise ValueError('handoff mismatch')
        for stage in ('stage1_lexical_50','stage1_lexical_100','stage1_lexical_150','stage1_lexical_200','stage1_lexical_400','stage2_joint_100'):
            if stage not in cfg['bank_diversity'] or stage not in cfg['song_diversity']: raise ValueError('missing entropy')
        if arm=='baseline':
            for stage in ('stage1_initial_0','stage1_lexical_400','stage2_initial_0','stage2_joint_100'):
                if snaps[stage]!=prior['snapshots'][stage]: raise ValueError('historical baseline mismatch')
        for stage in ('fixed_lexical400','fixed_joint100'):
            m=cfg[stage]; passed=bool(m['sense']>=10/12 and m['auc']>=.75 and m['gain']>=.03)
            if m['pass']!=passed: raise ValueError('incorrect pass')
        results[f'{perm}:{seed}:{arm}']={'run_id':run.info.run_id,'lexical400':cfg['fixed_lexical400'],'joint100':cfg['fixed_joint100'],
            'bank_diversity':cfg['bank_diversity'],'song_diversity':cfg['song_diversity'],
            'family':cfg['fixed_joint100_diagnosis']['family_summary'],
            'peak_bytes':cfg['peak_process_working_set_bytes'],'elapsed_seconds':cfg['elapsed_seconds']}
    counts={str(p):{a:sum(results[f'{p}:{s}:{a}']['joint100']['pass'] for s in SEEDS) for a in ARMS} for p in PERMUTATIONS}
    entropy_differences={str(p):[results[f'{p}:{s}:early-bank']['bank_diversity']['stage1_lexical_100']['entropy']-results[f'{p}:{s}:baseline']['bank_diversity']['stage1_lexical_100']['entropy'] for s in SEEDS] for p in PERMUTATIONS}
    medians={p:float(np.median(values)) for p,values in entropy_differences.items()}
    criteria={'practical_rescue':counts['7043']['early-bank']>=3 and counts['7019']['early-bank']==4 and counts['7027']['early-bank']>=2,
              'manipulation_7043':medians['7043']>=.10,
              'early_beats_late_7043':counts['7043']['early-bank']>=counts['7043']['late-bank']+1,
              'song_matches_early_7043':counts['7043']['early-song']>=counts['7043']['early-bank']}
    return {'runs':results,'interrupted_attempts':interrupted,'counts':counts,'paired_bank_entropy_differences_lexical100':entropy_differences,
            'median_paired_entropy_differences':medians,'frozen_criteria':criteria,
            'limitations':'exploratory selected corpus; bank targeted with prior knowledge, oracle-assisted toy; no test or hardware; no causal or generality claim'}


if __name__=='__main__':
    mlflow.set_tracking_uri(tracking_uri()); summary=inspect(); mlflow.set_experiment('word-energy-entropy-intervention-v1')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='ising_entropy_intervention_summary',energy_convention='beta E; corrected F3; temporary loss regularizer',variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',sampler_schedule='summary of exact enumeration training',seed=0,data_hash='see source runs',data_split='selected corpus 227 validation only',graph_summary='3 token star, 2 spins',limitations=summary['limitations'])
    with mlflow.start_run(run_name='entropy-intervention-panel-summary') as run:
        log_contract(mlflow,contract,summary)
        mlflow.log_metrics({f'perm{p}.{a}.pass_count':float(n) for p,arms in summary['counts'].items() for a,n in arms.items()})
        mlflow.log_metrics({f'criterion.{key}':float(v) for key,v in summary['frozen_criteria'].items()})
        print(json.dumps({'summary_run_id':run.info.run_id,'counts':summary['counts'],'entropy_medians':summary['median_paired_entropy_differences'],'criteria':summary['frozen_criteria']}))
