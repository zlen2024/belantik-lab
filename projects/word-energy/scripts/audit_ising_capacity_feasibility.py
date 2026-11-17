"""Posthoc constructive witness and context sanity checks; no model training."""
from pathlib import Path
import ast
import json
import sys
import time
from urllib.parse import urlparse
from urllib.request import url2pathname
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.diagnose_ising_validation import diagnose_raw
from scripts.diagnose_corpus227_permutation_dynamics import Exact, U, corpus_objects, objectives
from scripts.run_ising_corrected_joint import snapshot_hash
from word_energy.ising.exact_training import _free_energy_factory, _lexical_auc, _region_mass
from word_energy.ising.spectral_initializer import array_hash
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri

SEEDS=(251,257,263,269)
WITNESSES={251:'7ea1b55f9e2c497f9857369030f3eb13',257:'f14cb0a40282461da5b443edfdd9d9c3'}
SUMMARY='a405b027a563409288b7ace0e20b7196'


def artifact_dir(run):
    parsed=urlparse(run.info.artifact_uri)
    if parsed.scheme!='file': raise ValueError('only the verified local artifact store is supported')
    return Path(url2pathname(parsed.path))


def config_for(run_id,client):
    run=client.get_run(run_id)
    if run.info.status!='FINISHED': raise ValueError(f'run not finished: {run_id}')
    artifacts=artifact_dir(run); path=artifacts/'config.json'
    return json.loads(path.read_text(encoding='utf-8')),sha256_file(path),artifacts


def inference_signature(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    wanted={'_spin_states','_parameters','jax_nn_sigmoid','_free_energy_factory','_lexical_auc','_sense_accuracy_and_loss','_region_mass'}
    functions={node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in wanted}
    if set(functions)!=wanted: raise ValueError(f'missing inference functions: {path}')
    return functions


def context_control(raw,data,swapped=False,coupling=1.0):
    _,jnp,_,unary,center=_free_energy_factory(True)
    anchors={f:[int(t) for row,label in zip(data['train_windows'],data['train_families']) if label==f for t in row[1:]] for f in ('F','R')}
    q={f:np.mean([np.asarray(unary(raw,t)) for t in rows],axis=0) for f,rows in anchors.items()}
    region=q['F']>q['R']; equal=q['F']==q['R']
    rows=data['validation_windows']; families=data['validation_families']
    indices={f:[i for i,label in enumerate(families) if label==f] for f in ('F','R')}
    if len(indices['F'])!=len(indices['R']): raise ValueError('unequal families')
    partner={a:b for a,b in zip(indices['F'],indices['R'])}
    partner.update({b:a for a,b in zip(indices['F'],indices['R'])})
    credits={f:[] for f in ('F','R')}; gains=[]
    for family in ('F','R'):
        for i in indices[family]:
            window=rows[partner[i] if swapped else i]
            posterior=np.asarray(center(raw,jnp.asarray(window),coupling))
            prior=np.asarray(center(raw,jnp.asarray(window),0.0))
            mass=_region_mass(posterior,region,equal); base=_region_mass(prior,region,equal)
            correct=mass if family=='F' else 1-mass
            prior_correct=base if family=='F' else 1-base
            gains.append(correct-prior_correct)
            credits[family].append(1. if correct>.5 else .5 if correct==.5 else 0.)
    return {'macro_accuracy':float(np.mean([np.mean(credits[f]) for f in ('F','R')])),
            'correct_counts':{f:float(sum(credits[f])) for f in ('F','R')},
            'mean_gain':float(np.mean(gains)),'coupling':coupling,'swapped_context':swapped}


def frozen_bank_bound(raw,data):
    model=Exact(raw); prior=model.prob[data['ids']['bank']]; dominant=int(np.argmax(prior))
    log_ratios=np.log(prior/prior[dominant]) + np.abs(U-U[dominant])@model.g
    bound=float(1.0/(1.0+np.exp(np.delete(log_ratios,dominant)).sum()))
    _,jnp,_,_,center=_free_energy_factory(True)
    observed=[float(np.asarray(center(raw,jnp.asarray(row),1.0))[dominant])
              for row,f in zip(data['validation_windows'],data['validation_families']) if f in ('F','R')]
    if min(observed) < bound-1e-12: raise ValueError('frozen bank posterior bound violated')
    return {'dominant_state':U[dominant].tolist(),'dominant_prior':float(prior[dominant]),
            'lower_bound_any_two_contexts':bound,'minimum_observed_validation_posterior':min(observed)}


def audit():
    client=MlflowClient(); data=corpus_objects(); saved=json.loads((artifact_dir(client.get_run(SUMMARY))/'config.json').read_text(encoding='utf-8'))
    entries=[]; cached={}
    for perm in (7019,7043):
        for seed in SEEDS:
            run_id=saved['runs'][f'{perm}:{seed}']['run_id']
            cfg,source_hash,artifacts=config_for(run_id,client); cached[(perm,seed)]=cfg
            entries.append((f'perm{perm}',seed,run_id,cfg,source_hash,artifacts))
    for seed,run_id in WITNESSES.items():
        cfg,source_hash,artifacts=config_for(run_id,client)
        entries.append(('7043-early-bank',seed,run_id,cfg,source_hash,artifacts))
    result={}
    for name,seed,run_id,cfg,source_hash,artifacts in entries:
        expected=cached[(7043 if name=='7043-early-bank' else int(name[4:]),seed)]
        if cfg['replicate']!=227 or cfg['seed']!=seed or cfg['protocol_sha256']!=data['protocol_sha256'] or cfg['schedule_sha256']!=data['schedule_sha256']:
            raise ValueError(f'identity/protocol mismatch: {name}:{seed}')
        expected_perm=7043 if name=='7043-early-bank' else int(name[4:])
        expected_arm='early-bank' if name=='7043-early-bank' else 'permuted'
        if cfg['permutation_seed']!=expected_perm or cfg['arm']!=expected_arm or cfg['train_token_hash']!=array_hash(data['train_windows']) or cfg['pools']!={k:list(v) for k,v in data['pools'].items()}:
            raise ValueError(f'run identity/data mismatch: {name}:{seed}')
        sources=artifacts/'source_snapshot'
        for relative in ('word_energy/fresh_synthetic.py','scripts/diagnose_ising_validation.py'):
            local=ROOT/relative; historical=sources/local.name
            if sha256_file(local)!=sha256_file(historical): raise ValueError(f'source mismatch: {relative}')
        if inference_signature(sources/'exact_training.py')!=inference_signature(ROOT/'word_energy/ising/exact_training.py'):
            raise ValueError('inference code differs from source snapshot')
        if cfg['vocabulary']!=list(data['ids']) or not cfg['lexical_config']['normalize_spokes'] or not cfg['joint_config']['normalize_spokes']:
            raise ValueError(f'vocab/model mismatch: {name}:{seed}')
        if cfg['snapshots']['stage1_lexical_400']!=cfg['snapshots']['stage2_initial_0']:
            raise ValueError('handoff mismatch')
        if name=='7043-early-bank' and cfg['snapshots']['stage1_initial_0']!=expected['snapshots']['stage1_initial_0']:
            raise ValueError('same-initializer witness mismatch')
        raw={k:np.asarray(v,dtype=np.float64) for k,v in cfg['snapshots']['stage2_joint_100'].items()}
        if set(raw)!={'biases','internal','interaction'} or raw['biases'].shape!=(len(data['ids']),2) or raw['internal'].shape!=(len(data['ids']),) or raw['interaction'].shape!=(2,):
            raise ValueError('shape mismatch')
        if not all(np.isfinite(v).all() for v in raw.values()): raise ValueError('nonfinite parameter')
        checkpoint_hash=snapshot_hash(cfg['snapshots']['stage2_joint_100'])
        diag=diagnose_raw(raw,seed,'capacity-audit',checkpoint_hash,normalize_spokes=True,data_override=data)
        _,jnp,energy,_,_=_free_energy_factory(True)
        auc=_lexical_auc(raw,energy,data,jnp)
        gain=(diag['family_summary']['F']['gain']+diag['family_summary']['R']['gain'])/2
        score={'sense':diag['validation_macro_sense_accuracy'],'auc':auc,'gain':gain}
        for field in ('sense','auc','gain'):
            if not np.isclose(score[field],cfg['fixed_joint100'][field],atol=1e-12,rtol=0): raise ValueError(f'metric mismatch {name}:{seed}:{field}')
        _,lexical,joint,_,_,losses=objectives(raw,data)
        control=context_control(raw,data); zero=context_control(raw,data,coupling=0.0); swap=context_control(raw,data,swapped=True)
        if not np.isclose(control['macro_accuracy'],score['sense'],atol=1e-12) or not np.isclose(control['mean_gain'],gain,atol=1e-12): raise ValueError('context computation mismatch')
        if abs(zero['mean_gain'])>1e-12: raise ValueError('zero coupling sanity failed')
        result[f'{name}:{seed}']={'run_id':run_id,'source_config_sha256':source_hash,'checkpoint_sha256':checkpoint_hash,'original':score,
            'passed':bool(score['sense']>=10/12 and auc>=.75 and gain>=.03),
            'validation_family_summary':diag['family_summary'], 'zero_coupling':zero,'swapped_context':swap,
            'frozen_bank_bound':frozen_bank_bound(raw,data),
            'train_F2_loss':lexical,'train_F3_plus_quarter_F2_loss':joint,
            'train_cross_FR_negative_loss':float(losses[data['cross_fr_mask']].mean())}
    for seed in SEEDS:
        a,b=cached[(7019,seed)],cached[(7043,seed)]
        for key in ('internal','interaction'):
            if a['initial_hashes'][key]!=b['initial_hashes'][key]: raise ValueError('unpaired initial internal/interaction')
        for key in ('lexical_config','joint_config'):
            if a[key]!=b[key]: raise ValueError('unpaired training config')
    return {'protocol_sha256':data['protocol_sha256'],'schedule_sha256':data['schedule_sha256'],
            'checkpoints':result,'limitations':'Posthoc corpus227 validation only; train-anchor family labels and oracle negatives; no test, training, THRML or hardware; endpoint existence is not general capacity or optimizer proof.'}


if __name__=='__main__':
    start=time.monotonic(); mlflow.set_tracking_uri(tracking_uri()); result=audit()
    result['elapsed_seconds']=time.monotonic()-start
    mlflow.set_experiment('word-energy-capacity-feasibility-audit')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='ising_capacity_witness_audit',energy_convention='dimensionless beta E; corrected F3 lambda/2',variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',sampler_schedule='exact enumeration; checkpoint evaluation only',seed=0,data_hash=result['protocol_sha256'],data_split='posthoc corpus227 train and validation; no test',graph_summary='3 token star, 2 spins each',limitations=result['limitations'])
    with mlflow.start_run(run_name='corpus227-capacity-witness-controls') as run:
        log_contract(mlflow,contract,result)
        for path in (Path(__file__),ROOT/'notes/ISING_CAPACITY_FEASIBILITY_AUDIT_PROTOCOL.md',ROOT/'scripts/diagnose_ising_validation.py',ROOT/'scripts/diagnose_corpus227_permutation_dynamics.py',ROOT/'word_energy/ising/exact_training.py'):
            mlflow.log_artifact(str(path),'source_snapshot')
        mlflow.log_metrics({'checkpoint_count':len(result['checkpoints']),'witness_passes':sum(row['passed'] for row in result['checkpoints'].values())})
        print(json.dumps({'run_id':run.info.run_id,'elapsed_seconds':result['elapsed_seconds'],'checkpoints':{k:{'original':v['original'],'passed':v['passed'],'zero_gain':v['zero_coupling']['mean_gain'],'swapped_accuracy':v['swapped_context']['macro_accuracy'],'swapped_gain':v['swapped_context']['mean_gain'],'train_F2_loss':v['train_F2_loss'],'cross_FR_loss':v['train_cross_FR_negative_loss'],'frozen_bank_bound':v['frozen_bank_bound']['lower_bound_any_two_contexts']} for k,v in result['checkpoints'].items()}}))
