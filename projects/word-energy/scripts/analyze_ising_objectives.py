"""Evaluate objectives and fixed-batch gradients on saved snapshots only."""
from pathlib import Path
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from word_energy.ising.exact_training import _free_energy_factory, prepare_data
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri

if __name__=='__main__':
    seed=int(sys.argv[1]); branch=sys.argv[2]
    source=ROOT/'experiments'/'runs'/f'ising-attribution-{branch}-seed{seed}.json'
    payload=json.loads(source.read_text(encoding='utf-8')); data=prepare_data()
    jax,jnp,energy,_,_=_free_energy_factory()
    lp=jnp.asarray(data['train_pairs'][data['lexical_schedule'][0]])
    ln=jnp.asarray(data['train_negatives'][data['lexical_schedule'][0]])
    ids=data['joint_schedule'][0]; windows=jnp.asarray(data['train_windows'][ids]); corrupt=jnp.asarray(data['train_corruptions'][ids])
    vp=jnp.asarray(data['validation_pairs']); vn=jnp.asarray(data['validation_negatives'])
    def lexical(raw,pairs,neg):
        pos=jax.vmap(lambda row:energy(raw,row))(pairs)
        negatives=jax.vmap(lambda left,row:jax.vmap(lambda right:energy(raw,jnp.asarray([left,right],dtype=jnp.int32)))(row))(pairs[:,0],neg)
        return jnp.mean(jax.nn.softplus(1+pos[:,None]-negatives))
    def ranking(raw):
        pos=jax.vmap(lambda row:energy(raw,row))(windows)
        neg=jax.vmap(lambda two:jax.vmap(lambda row:energy(raw,row))(two))(corrupt)
        return jnp.mean(jax.nn.softplus(.25+pos[:,None]-neg))
    def objective(raw): return ranking(raw)+.25*lexical(raw,lp,ln)
    lex=jax.jit(lambda r:lexical(r,lp,ln)); vallex=jax.jit(lambda r:lexical(r,vp,vn)); rank=jax.jit(ranking)
    grad=jax.jit(jax.grad(objective)); results={}
    for label, values in payload['snapshots'].items():
        raw={k:jnp.asarray(v,dtype=jnp.float64) for k,v in values.items()}
        lv=float(lex(raw)); rv=float(rank(raw)); g=grad(raw)
        results[label]={'train_fixed_batch_lexical_loss':lv,'validation_lexical_loss':float(vallex(raw)),'train_fixed_batch_joint_ranking_loss':rv,'train_fixed_batch_joint_total':rv+.25*lv,'fixed_batch_joint_gradient_l2':float(np.sqrt(sum(float(jnp.sum(x*x)) for x in g.values()))),'gradient_l2_by_parameter':{k:float(jnp.linalg.norm(v)) for k,v in g.items()}}
    mlflow.set_tracking_uri(tracking_uri()); mlflow.set_experiment('word-energy-exact-training')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='ising_exact_attribution_objectives',energy_convention='dimensionless beta E; normalized Ising',variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',sampler_schedule='read-only autodiff on fixed first training batches and validation lexical data',seed=seed,data_hash=sha256_file(ROOT/'experiments'/'runs'/'synthetic-split-manifest.json'),data_split='train fixed batches and validation only',graph_summary='3 tokens, 2 spins each, star',limitations='Fixed-batch gradients are diagnostic probes, not historical optimizer gradients; no test or hardware.')
    with mlflow.start_run(run_name=f'ising-attribution-objectives-{branch}-seed{seed}') as run:
        log_contract(mlflow,contract,{'source_sha256':sha256_file(source),'fixed_batch':'lexical_schedule[0], joint_schedule[0]','snapshots':results})
        for label,values in results.items(): mlflow.log_metrics({f'{label}.{k}':v for k,v in values.items() if isinstance(v,float)})
        print(json.dumps({'run_id':run.info.run_id,'seed':seed,'branch':branch,'snapshots':len(results)}))
