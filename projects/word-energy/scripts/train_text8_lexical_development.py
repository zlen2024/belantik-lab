"""One bounded lexical-only run with shared, frozen text8 samples."""
from pathlib import Path
import json
import os
import sys
import tempfile
import time
from urllib.parse import urlparse
from urllib.request import url2pathname
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.run_ising_corrected_joint import peak_working_set
from scripts.diagnose_corpus227_permutation_dynamics import Exact
from word_energy.ising.exact_training import _imports
from word_energy.text8_development import sha_array
from word_energy.text8_lexical_models import ARMS,SEEDS,functions,initial_parameters,pair_metrics
from word_energy.tracking import RunContract,log_contract,tracking_uri

BATCH=256
VAL_BATCH=500
EPOCHS=3


def get_data(run_id):
    run=MlflowClient().get_run(run_id)
    if run.info.status!='FINISHED' or run.data.tags.get('model_family')!='text8_lexical_data':
        raise ValueError('dataset run not complete')
    uri=urlparse(run.info.artifact_uri)
    if uri.scheme!='file': raise ValueError('expected local artifact store')
    root=Path(url2pathname(uri.path))
    manifest=json.loads((root/'config.json').read_text(encoding='utf-8'))
    with np.load(root/'dataset'/'samples.npz',allow_pickle=False) as archive:
        arrays={k:archive[k] for k in archive.files}
    if set(arrays)!=set(manifest['hashes']): raise ValueError('sample keys differ')
    for key,value in arrays.items():
        if sha_array(value)!=manifest['hashes'][key]: raise ValueError(f'data hash mismatch {key}')
    return arrays,manifest


def validate_model(params,score_fn,loss_fn,arrays):
    pairs=arrays['validation_pairs']; noise=arrays['validation_noise']; pos=[]; neg=[]; losses=[]
    for start in range(0,len(pairs),VAL_BATCH):
        stop=min(start+VAL_BATCH,len(pairs)); p,n=score_fn(params,pairs[start:stop],noise[start:stop])
        # Match the five-negative training objective; all twenty still score AUC/MRR.
        value=loss_fn(params,pairs[start:stop],noise[start:stop,:5])
        pos.append(np.asarray(p)); neg.append(np.asarray(n)); losses.append((float(value),stop-start))
    positive=np.concatenate(pos); negative=np.concatenate(neg)
    metrics=pair_metrics(positive,negative,arrays['counts'][pairs[:,0]])
    metrics['objective_loss']=float(sum(value*size for value,size in losses)/len(pairs))
    return metrics


def snapshot(params,stage,temp_root):
    values={key:np.asarray(value) for key,value in params.items()}
    if not all(np.isfinite(v).all() for v in values.values()): raise FloatingPointError('nonfinite parameters')
    path=temp_root/f'{stage}.npz'; np.savez_compressed(path,**values)
    mlflow.log_artifact(str(path),'checkpoints')
    return {key:sha_array(value) for key,value in values.items()}


def parameter_diagnostics(params,arm):
    raw={k:np.asarray(v) for k,v in params.items()}
    if arm.startswith('ising'):
        model=Exact(raw); q=model.prob
        entropy=-(q*np.log(q)).sum(axis=1)
        return {'unary_entropy_mean':float(np.mean(entropy)),'unary_entropy_min':float(np.min(entropy)),
                'unary_max_probability_mean':float(np.max(q,axis=1).mean()),
                'unary_probability_above_099_fraction':float(np.mean(np.max(q,axis=1)>.99)),
                'bias_near_bound_fraction':float(np.mean(np.abs(model.b)>2.45)),
                'internal_near_bound_fraction':float(np.mean(np.abs(model.k)>1.96)),
                'coupling_0':float(model.g[0]),'coupling_1':float(model.g[1])}
    return {'input_norm_mean':float(np.linalg.norm(raw['input'],axis=1).mean()),
            'output_norm_mean':float(np.linalg.norm(raw['output'],axis=1).mean())}


def log_metrics(stage,metrics):
    scalar={f'{stage}.{key}':float(value) for key,value in metrics.items() if isinstance(value,(float,int))}
    scalar.update({f'{stage}.frequency_tercile_{i+1}_auc':float(value) for i,value in enumerate(metrics['auc_by_center_frequency_tercile'])})
    mlflow.log_metrics(scalar)


if __name__=='__main__':
    data_run,arm,seed=sys.argv[1],sys.argv[2],int(sys.argv[3])
    if arm not in ARMS or seed not in SEEDS: raise ValueError('outside frozen arm/seed set')
    start=time.monotonic(); mlflow.set_tracking_uri(tracking_uri()); arrays,manifest=get_data(data_run)
    exp=mlflow.set_experiment('word-energy-text8-lexical-development')
    existing=[r for r in MlflowClient().search_runs([exp.experiment_id],max_results=1000)
              if r.data.tags.get('arm')==arm and r.data.tags.get('seed')==str(seed)]
    if existing: raise ValueError(f'duplicate arm/seed, inspect prior attempt: {arm}/{seed}')
    jax,jnp,_=_imports(); score_fn,loss_fn,step,optimizer=functions(arm)
    params=initial_parameters(arm,seed,len(manifest['vocabulary'])); state=optimizer.init(params)
    train_pairs=arrays['shuffled_train_pairs'] if arm=='ising-shuffled' else arrays['train_pairs']
    vector=not arm.startswith('ising'); parameter_count=sum(np.asarray(x).size for x in params.values())
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',
        model_family='text8_'+arm,energy_convention='vector dot score' if vector else 'dimensionless beta E; negative normalized Ising F2 free energy',
        variable_encoding='2D vectors' if vector else 'spins {-1,+1}',beta_temperature_convention='not applicable' if vector else 'beta=1',
        sampler_schedule='classical exact enumeration for Ising; precomputed noise; Adam 3 epochs, batch 256',seed=seed,
        data_hash=manifest['source']['content_sha256'],data_split='train bytes 10–11M; validation 16–16.5M; reserved test not evaluated',
        graph_summary='two-table 2D embedding' if vector else 'two tokens x 2 spins; 4 potential edges, max degree 2',
        limitations='posthoc text8 development; no document boundaries, potential false negatives, no sense labels/test/THRML/hardware')
    details={'dataset_run_id':data_run,'dataset_hashes':manifest['hashes'],'arm':arm,'seed':seed,
             'parameter_count':parameter_count,'vocabulary_size':len(manifest['vocabulary']),'epochs':EPOCHS,
             'batch':BATCH,'learning_rate':.01,'gradient_clip':5.,'checkpoint_rule':'fixed epoch3',
             'thread_environment':{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},
             'train_samples_per_epoch':len(train_pairs),'validation_samples':len(arrays['validation_pairs']),
             'metrics':{},'checkpoint_hashes':{},'run_complete':False}
    with mlflow.start_run(run_name=f'text8-{arm}-seed{seed}') as run:
        mlflow.set_tags({'arm':arm,'seed':str(seed),'dataset_run_id':data_run,'test_status':'reserved not evaluated'})
        log_contract(mlflow,contract,details)
        mlflow.log_params({'arm':arm,'seed':seed,'epochs':EPOCHS,'batch':BATCH,'lr':.01,'clip':5.,'parameter_count':parameter_count,'dataset_run_id':data_run})
        for path in (Path(__file__),ROOT/'word_energy/text8_lexical_models.py',ROOT/'word_energy/text8_development.py',ROOT/'word_energy/ising/exact_training.py',ROOT/'scripts/diagnose_corpus227_permutation_dynamics.py',ROOT/'scripts/run_ising_corrected_joint.py',ROOT/'notes/TEXT8_LEXICAL_DEVELOPMENT_PROTOCOL.md'):
            mlflow.log_artifact(str(path),'source_snapshot')
        temp_root=ROOT/'.tmp'; temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as temporary:
            tmp=Path(temporary)
            details['checkpoint_hashes']['initial']=snapshot(params,'initial',tmp)
            details['metrics']['initial']=validate_model(params,score_fn,loss_fn,arrays)
            details['metrics']['initial'].update(parameter_diagnostics(params,arm))
            log_metrics('initial',details['metrics']['initial'])
            for epoch in range(EPOCHS):
                total=0.; count=0
                for i in range(0,len(train_pairs),BATCH):
                    ix=arrays['orders'][epoch,i:i+BATCH]
                    params,state,value=step(params,state,train_pairs[ix],arrays['train_noise'][epoch,ix])
                    value=float(value)
                    if not np.isfinite(value): raise FloatingPointError('nonfinite training loss')
                    total+=value*len(ix); count+=len(ix)
                stage=f'epoch{epoch+1}'
                peak=peak_working_set()
                if peak>2*1024**3: raise MemoryError('process exceeded 2 GiB peak working set')
                details['checkpoint_hashes'][stage]=snapshot(params,stage,tmp)
                metrics=validate_model(params,score_fn,loss_fn,arrays)
                metrics.update(parameter_diagnostics(params,arm))
                metrics['train_objective_mean']=total/count; metrics['peak_working_set_mib']=peak/1024**2
                details['metrics'][stage]=metrics
                log_metrics(stage,metrics)
                print(json.dumps({'run_id':run.info.run_id,'arm':arm,'seed':seed,'stage':stage,
                                  'noise_auc':metrics['noise_pairwise_auc'],'noise_mrr':metrics['noise_mrr_at_21'],
                                  'peak_mib':metrics['peak_working_set_mib']}),flush=True)
            details['elapsed_seconds']=time.monotonic()-start; details['run_complete']=True
            log_contract(mlflow,contract,details)
            mlflow.log_metric('resource.elapsed_seconds',details['elapsed_seconds'])
        print(json.dumps({'run_id':run.info.run_id,'arm':arm,'seed':seed,'status':'complete','elapsed_seconds':details['elapsed_seconds']}),flush=True)
