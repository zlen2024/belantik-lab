"""Compare learned pair Ising joint laws with THRML block Gibbs."""
from pathlib import Path
from itertools import product
import json,sys,tempfile,time
import numpy as np
from scipy.special import expit,logsumexp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.build_wikitext2_document_data import load
from word_energy.energy_embedding.mcmc_diagnostics import split_rank_rhat_ess
from word_energy.text8_development import sha_array
from word_energy.ising.exact_training import _imports
from word_energy.text8_factorial_models import score_pairs
from word_energy.tracking import RunContract,log_contract,tracking_uri
DATA='b37da8076fd54f4da389082278e9733a';MODEL='e6f37d07e0fa42d98185d5882ee50bb2'
STATES=np.array(list(product([-1.,1.],repeat=4)),dtype=np.float64)

def sample(bias,weights,seed):
    jax,jnp,_=_imports()
    from thrml import Block,SamplingSchedule,SpinNode,sample_states
    from thrml.models import IsingEBM,IsingSamplingProgram
    nodes=[SpinNode() for _ in range(4)];edges=[(nodes[0],nodes[1]),(nodes[2],nodes[3]),(nodes[0],nodes[2]),(nodes[1],nodes[3])]
    model=IsingEBM(nodes,edges,jnp.asarray(bias),jnp.asarray(weights),jnp.array(1.))
    blocks=[Block([nodes[0],nodes[3]]),Block([nodes[1],nodes[2]])];program=IsingSamplingProgram(model,blocks,[])
    schedule=SamplingSchedule(n_warmup=500,n_samples=2000,steps_per_sample=1)
    starts=jnp.array([[False,False,False,False],[True,True,True,True],[False,True,False,True],[True,False,True,False]],dtype=jnp.bool_)
    init=[starts[:,[0,3]],starts[:,[1,2]]]
    runner=jax.jit(jax.vmap(lambda ini,key:sample_states(key,program,schedule,ini,[],[Block(nodes)])))
    observed=runner(init,jax.random.split(jax.random.key(seed),4))[0]
    return 2*np.asarray(observed,dtype=np.int8)-1

def audit():
    client=MlflowClient();a,m=load(DATA);root=artifact_dir(client,MODEL);cfg=json.loads((root/'config.json').read_text(encoding='utf8'))
    if not cfg['run_complete'] or cfg['dataset_hashes']!=m['hashes'] or cfg['arm']!='role-frequency':raise ValueError('source run invalid')
    with np.load(root/'checkpoints/epoch3.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    if {k:sha_array(v) for k,v in raw.items()}!=cfg['checkpoint_hashes']['epoch3']:raise ValueError('checkpoint hash')
    vocab={w:i for i,w in enumerate(m['vocabulary'])};results={};samples_saved={}
    for word in ('river','money'):
        u,v=vocab['bank'],vocab[word];b=np.r_[2.5*np.tanh(raw['target_biases'][u]),2.5*np.tanh(raw['context_biases'][v])]
        weights=np.r_[2*np.tanh(raw['target_internal'][u]),2*np.tanh(raw['context_internal'][v]),1.5*expit(raw['interaction'])]
        logw=STATES@b+weights[0]*STATES[:,0]*STATES[:,1]+weights[1]*STATES[:,2]*STATES[:,3]+weights[2]*STATES[:,0]*STATES[:,2]+weights[3]*STATES[:,1]*STATES[:,3]
        prob=np.exp(logw-logsumexp(logw));left=STATES[:4,:2];single=np.array(list(product([-1.,1.],repeat=2)))
        logzl=logsumexp(single@b[:2]+weights[0]*single[:,0]*single[:,1]);logzr=logsumexp(single@b[2:]+weights[1]*single[:,0]*single[:,1])
        exact_score=float(logsumexp(logw)-logzl-logzr)
        code_score=float(np.asarray(score_pairs(raw,np.array([[u,v]],np.int32),np.zeros(len(vocab)),role_specific=True,with_frequency=False))[0])
        if abs(code_score-exact_score)>1e-12:raise ValueError('exact F2 scorer mismatch')
        rows=[]
        for seed in (17,29,43):
            x=sample(b,weights,seed);samples_saved[f'{word}_{seed}']=x
            counts=np.array([np.sum(np.all(x==s,axis=-1)) for s in STATES]);emp=counts/counts.sum();observables={}
            for i in range(4):
                vals=x[:,:,i].astype(float);reference=float(prob@STATES[:,i]);rh,ess,mcse=split_rank_rhat_ess(vals);observed=float(vals.mean())
                finite=bool(np.isfinite(rh) and np.isfinite(mcse));passed=bool(finite and rh<1.05 and ess>=400 and mcse<=.01 and abs(observed-reference)<=max(.02,3*mcse))
                observables[f'spin{i}']={'sample':observed,'exact':reference,'rhat':float(rh) if finite else None,'ess':float(ess),'mcse':float(mcse) if finite else None,'pass':passed}
            vals=np.all(x[:,:,:2]==1,axis=-1).astype(float);reference=float(prob[np.all(STATES[:,:2]==1,axis=1)].sum());rh,ess,mcse=split_rank_rhat_ess(vals);finite=bool(np.isfinite(rh) and np.isfinite(mcse));observed=float(vals.mean())
            observables['target_plusplus']={'sample':observed,'exact':reference,'rhat':float(rh) if finite else None,'ess':float(ess),'mcse':float(mcse) if finite else None,'pass':bool(finite and rh<1.05 and ess>=400 and mcse<=.01 and abs(observed-reference)<=max(.02,3*mcse))}
            rows.append({'seed':seed,'state_probability':emp.tolist(),'total_variation':float(np.abs(emp-prob).sum()/2),'observables':observables,'all_observables_pass':all(o['pass'] for o in observables.values())})
        results[word]={'target_id':u,'context_id':v,'bias':b.tolist(),'weights':weights.tolist(),'exact_state_probability':prob.tolist(),'exact_normalized_pair_score':exact_score,'source_score':code_score,'runs':rows}
    return {'dataset_run_id':DATA,'source_model_run':MODEL,'source_checkpoint_hashes':cfg['checkpoint_hashes']['epoch3'],'state_order':STATES.tolist(),'results':results,'limitations':'THRML simulator CPU; joint-state law only, not physical hardware or direct F2/free-energy estimate; host frequency prior not implemented by sampler'},samples_saved

if __name__=='__main__':
    start=time.monotonic();mlflow.set_tracking_uri(tracking_uri());result,samples=audit();result['elapsed_seconds']=time.monotonic()-start
    mlflow.set_experiment('word-energy-thrml-learned-pair-parity')
    contract=RunContract(execution_tier='thrml_simulator',device_class='cpu_gpu_simulator',model_family='wikitext_learned_pair_ising',energy_convention='E=-sum b*s-sum J*s*s, beta=1',variable_encoding='THRML bool True=+1',beta_temperature_convention='beta=1',sampler_schedule='block Gibbs two colors, warmup500, draws2000, step1',seed=17,data_hash=DATA,data_split='WikiText2 train-trained checkpoint; two illustrative vocab pairs; no test',graph_summary='4 spins, 4 edges, max degree2',limitations=result['limitations'],chain_count=4,burn_in=500,sample_count=2000)
    with mlflow.start_run(run_name='learned-bank-pair-four-chain-parity') as run:
        log_contract(mlflow,contract,result);tmp=ROOT/'.tmp';tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as d:
            p=Path(d)/'spin_samples.npz';np.savez_compressed(p,**samples);mlflow.log_artifact(str(p),'samples')
        for path in (Path(__file__),ROOT/'notes/THRML_LEARNED_PAIR_PROTOCOL.md'):mlflow.log_artifact(str(path),'source_snapshot')
        print(json.dumps({'run_id':run.info.run_id,'results':result['results']},indent=2))
