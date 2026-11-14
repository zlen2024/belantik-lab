"""Run the preregistered lexical-only bank initialization transplant."""
from pathlib import Path
import hashlib
import json
import sys
import time
import os
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from scripts.diagnose_ising_validation import diagnose_raw
from word_energy.ising.exact_training import ExactTrainingConfig, _free_energy_factory, _lexical_auc, prepare_data, token_index, train_smoke
from word_energy.synthetic import POOLS
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri

if __name__=='__main__':
    background=int(sys.argv[1]); donor=int(sys.argv[2]); block=sys.argv[3] if len(sys.argv)>3 else 'bank'
    if block not in ('bank','anchors','anchor_biases'):
        raise ValueError('block must be bank, anchors, or anchor_biases')
    if (background,donor) not in ((29,43),(43,29)):
        raise ValueError('only preregistered reciprocal swaps are permitted')
    def load(seed):
        path=ROOT/'experiments'/'runs'/f'ising-attribution-original-seed{seed}.json'
        payload=json.loads(path.read_text(encoding='utf-8'))
        if not payload['reproduction_pass']: raise ValueError('original replay failed')
        return path,{k:np.asarray(v,dtype=np.float64) for k,v in payload['snapshots']['initial_0'].items()}
    backpath, raw=load(background); donorpath, other=load(donor)
    ids=token_index(); bank=ids['bank']; original={k:v.copy() for k,v in raw.items()}
    selected=[bank] if block=='bank' else [ids[token] for family in ('F','R') for token in POOLS[family]]
    raw['biases'][selected]=other['biases'][selected]
    if block!='anchor_biases': raw['internal'][selected]=other['internal'][selected]
    for key in raw:
        if key=='interaction' or (key=='internal' and block=='anchor_biases'): assert np.array_equal(raw[key],original[key])
        else:
            mask=~np.isin(np.arange(len(raw[key])),selected)
            assert np.array_equal(raw[key][mask],original[key][mask])
    assert np.array_equal(raw['biases'][selected],other['biases'][selected])
    if block!='anchor_biases': assert np.array_equal(raw['internal'][selected],other['internal'][selected])
    snapshots={}
    def capture(stage, step, values):
        if not all(np.isfinite(v).all() for v in values.values()): raise FloatingPointError(f'{stage} {step}')
        snapshots[f'{stage}_{step}']={k:v.tolist() for k,v in values.items()}
    start=time.monotonic()
    _,metrics=train_smoke(ExactTrainingConfig(seed=background,joint_updates=0),snapshot_callback=capture,initial_override=raw)
    elapsed=time.monotonic()-start
    fixed={k:np.asarray(v,dtype=np.float64) for k,v in snapshots['lexical_400'].items()}
    digest=hashlib.sha256(json.dumps(snapshots['lexical_400'],sort_keys=True).encode()).hexdigest()
    diagnosis=diagnose_raw(fixed,background,'transplant-fixed-lexical400',digest)
    jax,jnp,energy,_,_=_free_energy_factory(); data=prepare_data()
    pairs=jnp.asarray(data['validation_pairs']); neg=jnp.asarray(data['validation_negatives'])
    positives=jax.vmap(lambda pair:energy(fixed,pair))(pairs)
    negatives=jax.vmap(lambda left,row:jax.vmap(lambda right:energy(fixed,jnp.asarray([left,right],dtype=jnp.int32)))(row))(pairs[:,0],neg)
    diagnosis['validation_lexical_loss']=float(jnp.mean(jax.nn.softplus(1+positives[:,None]-negatives)))
    diagnosis['validation_lexical_macro_auc']=_lexical_auc(fixed,energy,data,jnp)
    selected={k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in metrics.items()}
    payload={'background_seed':background,'donor_seed':donor,'transplant_block':block,'source_background_sha256':sha256_file(backpath),'source_donor_sha256':sha256_file(donorpath),'runner_sha256':sha256_file(Path(__file__).resolve()),'training_source_sha256':sha256_file(ROOT/'word_energy'/'ising'/'exact_training.py'),'thread_settings':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},'train_steps':400,'elapsed_seconds':elapsed,'selected_metrics':selected,'fixed_400_diagnosis':diagnosis,'snapshots':snapshots}
    mlflow.set_tracking_uri(tracking_uri()); mlflow.set_experiment('word-energy-exact-training')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family=f'ising_{block}_initialization_transplant',energy_convention='dimensionless beta E; normalized Ising',variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',sampler_schedule='exact enumeration; 400 lexical optimizer steps',seed=background,data_hash=sha256_file(ROOT/'experiments'/'runs'/'synthetic-split-manifest.json'),data_split='training and validation only',graph_summary='2 spins per word, lexical pairs and joint diagnostic window',limitations='Two reciprocal swaps on two synthetic seeds; fixed endpoint exploratory; no joint training or hardware.')
    with mlflow.start_run(run_name=f'ising-{block}-transplant-background{background}-donor{donor}') as run:
        log_contract(mlflow,contract,payload)
        mlflow.log_params({'background_seed':background,'donor_seed':donor,'transplant_block':block,'training_updates':400,'endpoint':'fixed_lexical400'})
        mlflow.log_metrics({'anchor_tv':diagnosis['anchor_total_variation'],'context_tv':diagnosis['context_posterior_total_variation'],'validation_sense_accuracy':diagnosis['validation_macro_sense_accuracy'],'validation_lexical_loss':diagnosis['validation_lexical_loss'],'validation_lexical_macro_auc':diagnosis['validation_lexical_macro_auc'],'elapsed_seconds':elapsed})
        print(json.dumps({'run_id':run.info.run_id,'background':background,'donor':donor,'anchor_tv':diagnosis['anchor_total_variation'],'context_tv':diagnosis['context_posterior_total_variation'],'sense':diagnosis['validation_macro_sense_accuracy'],'lexical_auc':diagnosis['validation_lexical_macro_auc'],'elapsed_seconds':elapsed}))
