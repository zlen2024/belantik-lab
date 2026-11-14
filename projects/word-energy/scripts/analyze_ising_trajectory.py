"""Analyze stored attribution snapshots with independent NumPy enumeration."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mlflow
from word_energy.ising.exact_training import prepare_data
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri

STATES = np.array([[-1,-1],[1,-1],[-1,1],[1,1]], dtype=np.float64)

def snapshot_stats(raw, data):
    b=2.5*np.tanh(np.asarray(raw['biases']))
    j=2*np.tanh(np.asarray(raw['internal']))
    w=1.5/(1+np.exp(-np.asarray(raw['interaction'])))
    logits=b@STATES.T+j[:,None]*(STATES[:,0]*STATES[:,1])
    logits-=logits.max(axis=1,keepdims=True)
    p=np.exp(logits); p/=p.sum(axis=1,keepdims=True)
    groups={f:[int(t) for row,family in zip(data['train_windows'],data['train_families']) if family==f for t in row[1:]] for f in ('F','R')}
    q={f:p[ids].mean(axis=0) for f,ids in groups.items()}
    region=q['F']>q['R']; equal=q['F']==q['R']
    post={'F':[],'R':[]}; rows=[]
    for idx,(tokens,family) in enumerate(zip(data['validation_windows'],data['validation_families'])):
        if family not in post: continue
        center=int(tokens[0]); mass=p[center].copy()
        for token in tokens[1:]:
            factor=np.exp((STATES[:,None,:]*STATES[None,:,:]*w).sum(axis=2))
            mass*=factor@p[int(token)]
        mass/=mass.sum(); post[family].append(mass)
        fmass=float(mass[region].sum()+0.5*mass[equal].sum())
        correct=fmass if family=='F' else 1-fmass
        prior=float(p[center,region].sum()+0.5*p[center,equal].sum())
        prior=prior if family=='F' else 1-prior
        rows.append({'row':idx,'family':family,'posterior':mass.tolist(),'correct_mass':correct,'prior_correct_mass':prior,'gain':correct-prior,'credit':1.0 if correct>.5 else .5 if correct==.5 else 0.0})
    return {'anchor_q':{f:q[f].tolist() for f in q},'anchor_tv':float(np.abs(q['F']-q['R']).sum()/2),'anchor_entropy':{f:float(-np.sum(q[f]*np.log(q[f]))) for f in q},'bank_prior':p[data['ids']['bank']].tolist(),'context_tv':float(np.abs(np.mean(post['F'],axis=0)-np.mean(post['R'],axis=0)).sum()/2),'family':{f:{k:float(np.mean([r[k] for r in rows if r['family']==f])) for k in ('correct_mass','prior_correct_mass','gain','credit')} for f in ('F','R')},'rows':rows,'parameter_saturation':{'max_abs_bias_fraction':float(np.max(np.abs(np.tanh(np.asarray(raw['biases']))))),'max_abs_internal_fraction':float(np.max(np.abs(np.tanh(np.asarray(raw['internal']))))),'interaction':w.tolist()}}

if __name__=='__main__':
    seed=int(sys.argv[1]); branch=sys.argv[2]
    source=ROOT/'experiments'/'runs'/f'ising-attribution-{branch}-seed{seed}.json'
    payload=json.loads(source.read_text(encoding='utf-8')); data=prepare_data()
    snapshots={label:snapshot_stats(raw,data) for label,raw in payload['snapshots'].items()}
    mlflow.set_tracking_uri(tracking_uri()); mlflow.set_experiment('word-energy-exact-training')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='ising_exact_attribution_trajectory',energy_convention='dimensionless beta E; normalized Ising',variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',sampler_schedule='read-only exact enumeration of saved snapshots',seed=seed,data_hash=sha256_file(ROOT/'experiments'/'runs'/'synthetic-split-manifest.json'),data_split='training anchors and validation only',graph_summary='3 tokens, 2 spins each, star',limitations='Exploratory validation; no new training or hardware; loss gradients and per-anchor details not computed.')
    with mlflow.start_run(run_name=f'ising-attribution-trajectory-{branch}-seed{seed}') as run:
        log_contract(mlflow,contract,{'source_sha256':sha256_file(source),'branch':branch,'snapshots':snapshots})
        for label,s in snapshots.items():
            mlflow.log_metrics({f'{label}.anchor_tv':s['anchor_tv'],f'{label}.context_tv':s['context_tv'],f'{label}.F.credit':s['family']['F']['credit'],f'{label}.R.credit':s['family']['R']['credit']})
        print(json.dumps({'run_id':run.info.run_id,'seed':seed,'branch':branch,'snapshots':len(snapshots),'initial':snapshots['initial_0']['anchor_tv'],'end':snapshots['joint_100']['anchor_tv']}))
