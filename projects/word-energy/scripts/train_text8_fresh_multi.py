"""Train one fixed-prior vector or new-permutation Ising development run."""
from pathlib import Path
import json,os,sys,tempfile,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.build_text8_fresh_multi_data import load,BASE
from scripts.train_text8_lexical_development import snapshot,log_metrics
from scripts.train_text8_frequency_shuffle import val_metrics,zero_control
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.run_ising_corrected_joint import peak_working_set
from word_energy.ising.exact_training import _imports
from word_energy.text8_canonical_eval import evaluate
from word_energy.text8_development import sha_array
from word_energy.text8_factorial_models import functions as ising_functions,initial as ising_initial
from word_energy.text8_prior_vector import functions as vector_functions,initial as vector_initial
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file

SEEDS=(17,29,43); MODELS=('tied-frequency','role-frequency','vector-frequency'); PERMS=(11330,11331,11332)
if __name__=='__main__':
    data_id,arm,seed,shuffle=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
    if arm not in MODELS or seed not in SEEDS or shuffle not in ((0,*PERMS) if arm=='vector-frequency' else (11331,11332)): raise ValueError('outside frozen panel')
    started=time.monotonic();mlflow.set_tracking_uri(tracking_uri()); base,fresh,meta=load(data_id)
    arrays=dict(base);arrays.update({'validation_pairs':fresh['validation_pairs'],'validation_noise':fresh['validation_noise']})
    train=base['train_pairs'] if shuffle==0 else fresh[f'pairs_{shuffle}']
    offset=.25*np.log(base['counts'].astype(np.float64));jax,jnp,_=_imports()
    vector=arm=='vector-frequency'
    if vector:
        params=vector_initial(seed,len(offset));score,loss,step,optimizer=vector_functions(offset)
    else:
        params=ising_initial(arm,seed,len(offset));score,loss,step,optimizer=ising_functions(arm,offset)
    state=optimizer.init(params); count=sum(np.asarray(x).size for x in params.values())
    prior_id=None
    if not vector:
        client=MlflowClient(); exp=client.get_experiment_by_name('word-energy-text8-frequency-role-factorial')
        old=[r for r in client.search_runs([exp.experiment_id],max_results=1000) if r.data.tags.get('arm')==arm and r.data.tags.get('seed')==str(seed)]
        if len(old)!=1 or old[0].info.status!='FINISHED': raise ValueError('missing original')
        prior_id=old[0].info.run_id;root=artifact_dir(client,prior_id);config=json.loads((root/'config.json').read_text(encoding='utf-8'))
        if not config['run_complete'] or config['dataset_hashes']!=meta['base_array_hashes'] or config['offset_sha256']!=sha_array(offset) or config['checkpoint_rule']!='fixed epoch3': raise ValueError('original source config mismatch')
        if config['checkpoint_hashes']['initial']!={k:sha_array(np.asarray(v)) for k,v in params.items()}: raise ValueError('original initialization mismatch')
        for rel in ('word_energy/text8_factorial_models.py','word_energy/text8_canonical_eval.py','word_energy/text8_lexical_models.py','word_energy/ising/exact_training.py','scripts/train_text8_lexical_development.py','word_energy/text8_development.py'):
            p=ROOT/rel
            if sha256_file(p)!=sha256_file(root/'source_snapshot'/p.name): raise ValueError('original source mismatch')
    experiment=mlflow.set_experiment('word-energy-text8-fresh-multi-shuffle')
    existing=[r for r in MlflowClient().search_runs([experiment.experiment_id],max_results=1000) if r.data.tags.get('arm')==arm and r.data.tags.get('seed')==str(seed) and r.data.tags.get('shuffle')==str(shuffle)]
    if existing: raise ValueError('duplicate run; inspect failure')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='text8_fresh_multi_'+arm,
      energy_convention='2D dot + host frequency offset' if vector else 'dimensionless normalized F2 + host frequency offset',
      variable_encoding='2D real vectors' if vector else 'spins {-1,+1}',beta_temperature_convention='not applicable' if vector else 'beta=1',
      sampler_schedule='classical exact scoring; fixed noise; Adam 3 epochs batch256',seed=seed,data_hash=meta['source']['content_sha256'],
      data_split='train 10-11M, fresh validation 17-17.5M, reserved test untouched',graph_summary='directed 2D vectors' if vector else '4 spins, 4 edges, degree 2',
      limitations='posthoc development; incomplete source provenance, shared byte corpus/no boundaries; no sense labels or hardware')
    details={'fresh_data_run':data_id,'base_data_run':BASE,'base_hashes':meta['base_array_hashes'],'fresh_hashes':meta['hashes'],'paired_original_run_id':prior_id,
      'arm':arm,'seed':seed,'shuffle_seed':shuffle,'train_pair_hash':sha_array(train),'offset_sha256':sha_array(offset),'parameter_count':count,
      'epochs':3,'batch':256,'lr':.01,'clip':5.,'checkpoint_rule':'epoch3','thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},
      'run_complete':False,'metrics':{},'checkpoint_hashes':{}}
    with mlflow.start_run(run_name=f'fresh-{arm}-seed{seed}-shuffle{shuffle}') as run:
        mlflow.set_tags({'arm':arm,'seed':str(seed),'shuffle':str(shuffle),'fresh_data_run':data_id,'test_status':'reserved not evaluated'})
        mlflow.log_params({'arm':arm,'seed':seed,'shuffle_seed':shuffle,'parameter_count':count,'epochs':3,'batch':256,'learning_rate':.01,'clip':5.,'fresh_data_run':data_id})
        log_contract(mlflow,contract,details)
        for rel in ('scripts/train_text8_fresh_multi.py','scripts/build_text8_fresh_multi_data.py','scripts/run_text8_fresh_multi.py','word_energy/text8_prior_vector.py','word_energy/text8_factorial_models.py','word_energy/text8_canonical_eval.py','word_energy/text8_lexical_models.py','word_energy/ising/exact_training.py','scripts/train_text8_frequency_shuffle.py','scripts/train_text8_lexical_development.py','word_energy/text8_development.py','notes/TEXT8_FRESH_MULTI_SHUFFLE_PROTOCOL.md','tests/test_text8_prior_vector.py'):
            mlflow.log_artifact(str(ROOT/rel),'source_snapshot')
        if vector:
            zero={k:jnp.zeros_like(v) for k,v in params.items()};control=evaluate(zero,score,arrays['validation_pairs'],arrays['validation_noise'],arrays['counts'][arrays['validation_pairs'][:,0]],symmetric=False)
        else: control=zero_control(params,arm,offset,arrays,meta['frequency_only'])
        for key in ('noise_pairwise_auc','noise_mrr_at_21'):
            if not np.isclose(control[key],meta['frequency_only'][key],rtol=0,atol=1e-14): raise ValueError('frequency baseline mismatch')
        details['zero_control']=control
        tmp=ROOT/'.tmp';tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as d:
            tmp=Path(d);details['checkpoint_hashes']['initial']=snapshot(params,'initial',tmp)
            details['metrics']['initial']=val_metrics(params,arm,score,loss,arrays);log_metrics('initial',details['metrics']['initial'])
            for epoch in range(3):
                total=0.;n=0
                for i in range(0,len(train),256):
                    ix=base['orders'][epoch,i:i+256];params,state,value=step(params,state,train[ix],base['train_noise'][epoch,ix]);value=float(value)
                    if not np.isfinite(value): raise FloatingPointError('nonfinite loss')
                    total+=value*len(ix);n+=len(ix)
                peak=peak_working_set()
                if peak>2*1024**3: raise MemoryError('2GiB exceeded')
                stage=f'epoch{epoch+1}';details['checkpoint_hashes'][stage]=snapshot(params,stage,tmp)
                metrics=val_metrics(params,arm,score,loss,arrays);metrics['train_objective_mean']=total/n;metrics['peak_working_set_mib']=peak/1024**2
                details['metrics'][stage]=metrics;log_metrics(stage,metrics)
                print(json.dumps({'run_id':run.info.run_id,'arm':arm,'seed':seed,'shuffle':shuffle,'stage':stage,'auc':metrics['noise_pairwise_auc'],'mrr':metrics['noise_mrr_at_21']}),flush=True)
            details['elapsed_seconds']=time.monotonic()-started;details['run_complete']=True;log_contract(mlflow,contract,details);mlflow.log_metric('resource.elapsed_seconds',details['elapsed_seconds'])
