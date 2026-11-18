"""One paired context-shuffled development control with frequency offset."""
from pathlib import Path
import json
import os
import sys
import tempfile
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.run_ising_corrected_joint import peak_working_set
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.train_text8_lexical_development import get_data,snapshot,log_metrics
from word_energy.ising.exact_training import _imports
from word_energy.text8_canonical_eval import evaluate
from word_energy.text8_development import sha_array
from word_energy.text8_factorial_models import ARMS,SEEDS,functions,initial,score_pairs
from word_energy.text8_lexical_models import pair_metrics
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file

DATA_RUN='6cf76591c3fa49c686a3d81d7f225607'
BATCH=256


def val_metrics(params,arm,score_fn,loss_fn,arrays):
    pairs=arrays['validation_pairs']; noise=arrays['validation_noise']
    metrics=evaluate(params,score_fn,pairs,noise,arrays['counts'][pairs[:,0]],symmetric=(arm=='tied'))
    weighted=0.
    for i in range(0,len(pairs),500):
        stop=min(i+500,len(pairs)); weighted+=float(loss_fn(params,pairs[i:stop],noise[i:stop,:5]))*(stop-i)
    metrics['objective_loss']=weighted/len(pairs)
    return metrics


def zero_control(params,arm,offset,arrays,frequency_expected):
    jax,jnp,_=_imports(); pairs=arrays['validation_pairs']; noise=arrays['validation_noise']
    role=arm.startswith('role'); with_frequency=arm.endswith('frequency')
    @jax.jit
    def zero_score(p):
        return score_pairs(params,p,jnp.asarray(offset),role_specific=role,with_frequency=with_frequency,coupling_scale=0.0)
    positive=np.asarray(zero_score(pairs))
    negative_pairs=np.stack((np.broadcast_to(pairs[:,0,None],noise.shape),noise),axis=-1).reshape((-1,2))
    negative=np.asarray(zero_score(negative_pairs)).reshape(noise.shape)
    expected_positive=offset[pairs[:,1]] if with_frequency else np.zeros(len(pairs))
    expected_negative=offset[noise] if with_frequency else np.zeros(noise.shape)
    if not np.array_equal(positive,expected_positive) or not np.array_equal(negative,expected_negative):
        raise ValueError('zero coupling score mismatch')
    metrics=pair_metrics(positive,negative,arrays['counts'][pairs[:,0]])
    expected=frequency_expected if with_frequency else {'noise_pairwise_auc':.5,'noise_mrr_at_21':sum(1/i for i in range(1,22))/21}
    for key in ('noise_pairwise_auc','noise_mrr_at_21'):
        if not np.isclose(metrics[key],expected[key],atol=1e-14,rtol=0): raise ValueError('zero coupling metric mismatch')
    return metrics


if __name__=='__main__':
    arm,seed=sys.argv[1],int(sys.argv[2])
    if arm not in ('tied-frequency','role-frequency'): raise ValueError('expected frequency arm')
    if arm not in ARMS or seed not in SEEDS: raise ValueError('outside frozen panel')
    start=time.monotonic(); mlflow.set_tracking_uri(tracking_uri()); arrays,manifest=get_data(DATA_RUN)
    experiment=mlflow.set_experiment('word-energy-text8-frequency-shuffle-control')
    original=arrays['train_pairs']; shuffled=arrays['shuffled_train_pairs']; order=arrays['shuffle_order']
    if not np.array_equal(np.sort(order),np.arange(len(order))) or not np.array_equal(shuffled[:,0],original[:,0]) or not np.array_equal(shuffled[:,1],original[order,1]): raise ValueError('invalid context permutation')
    existing=[r for r in MlflowClient().search_runs([experiment.experiment_id],max_results=1000) if r.data.tags.get('arm')==arm and r.data.tags.get('seed')==str(seed)]
    if existing: raise ValueError('existing arm/seed; inspect before retry')
    jax,jnp,_=_imports(); offset=.25*np.log(arrays['counts'].astype(np.float64))
    score_fn,loss_fn,step,optimizer=functions(arm,offset)
    params=initial(arm,seed,len(offset)); state=optimizer.init(params)
    role=arm.startswith('role'); frequency=arm.endswith('frequency')
    count=sum(np.asarray(v).size for v in params.values())
    client=MlflowClient(); prior_exp=client.get_experiment_by_name('word-energy-text8-frequency-role-factorial')
    matches=[r for r in client.search_runs([prior_exp.experiment_id],max_results=1000) if r.data.tags.get('arm')==arm and r.data.tags.get('seed')==str(seed)]
    if len(matches)!=1 or matches[0].info.status!='FINISHED': raise ValueError('missing unique paired original')
    prior=matches[0]; prior_root=artifact_dir(client,prior.info.run_id)
    prior_config=json.loads((prior_root/'config.json').read_text(encoding='utf-8'))
    if not prior_config['run_complete'] or prior_config['dataset_hashes']!=manifest['hashes'] or prior_config['offset_sha256']!=sha_array(offset): raise ValueError('paired data mismatch')
    if prior_config['checkpoint_hashes']['initial']!={k:sha_array(np.asarray(v)) for k,v in params.items()}: raise ValueError('paired initialization mismatch')
    for rel in ('word_energy/text8_factorial_models.py','word_energy/text8_canonical_eval.py','word_energy/text8_lexical_models.py','word_energy/ising/exact_training.py','scripts/train_text8_lexical_development.py','word_energy/text8_development.py'):
        source=ROOT/rel
        if sha256_file(source)!=sha256_file(prior_root/'source_snapshot'/source.name): raise ValueError('paired source mismatch '+rel)
    diagnostics={'fixed_shuffle_indices':int(np.count_nonzero(order==np.arange(len(order)))),
      'equal_context_rows':int(np.count_nonzero(original[:,1]==shuffled[:,1])),
      'distinct_pairs_original':len(np.unique(original,axis=0)),'distinct_pairs_shuffled':len(np.unique(shuffled,axis=0)),
      'self_id_original':int(np.count_nonzero(original[:,0]==original[:,1])), 'self_id_shuffled':int(np.count_nonzero(shuffled[:,0]==shuffled[:,1])),
      'positive_noise_collisions_original':[int(np.sum(original[:,1,None]==x)) for x in arrays['train_noise']],
      'positive_noise_collisions_shuffled':[int(np.sum(shuffled[:,1,None]==x)) for x in arrays['train_noise']]}
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='text8_frequency_shuffled_'+arm,
        energy_convention='dimensionless beta E normalized F2; '+('fixed candidate log-frequency host readout' if frequency else 'no host offset'),
        variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',
        sampler_schedule='exact 4-spin F2; precomputed noise; Adam 3 epochs, batch 256',seed=seed,
        data_hash=manifest['source']['content_sha256'],data_split='existing text8 development train/validation; reserved test not evaluated',
        graph_summary='4-spin F2, 4 potential edges, max degree 2; '+('separate role tables' if role else 'shared word table'),
        limitations='posthoc development; one fixed context permutation; offset host readout; no sense labels/test/THRML/hardware')
    details={'dataset_run_id':DATA_RUN,'dataset_hashes':manifest['hashes'],'paired_original_run_id':prior.info.run_id,'shuffle_diagnostics':diagnostics,'arm':arm,'seed':seed,
             'parameter_count':count,'vocabulary_size':len(offset),'epochs':3,'batch':BATCH,'learning_rate':.01,'gradient_clip':5.,
             'frequency_offset_formula':'.25*log(train_count)' if frequency else 'none',
             'offset_sha256':sha_array(offset),'training_pairs':'shuffled_train_pairs','shuffle_seed':11330,'checkpoint_rule':'fixed epoch3',
             'thread_environment':{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},
             'metrics':{},'checkpoint_hashes':{},'run_complete':False}
    with mlflow.start_run(run_name=f'text8-shuffle-{arm}-seed{seed}') as run:
        mlflow.set_tags({'arm':arm,'seed':str(seed),'paired_original_run_id':prior.info.run_id,'dataset_run_id':DATA_RUN,'test_status':'reserved not evaluated'})
        log_contract(mlflow,contract,details)
        mlflow.log_params({'arm':arm,'seed':seed,'epochs':3,'batch':BATCH,'lr':.01,'clip':5.,'parameter_count':count,'dataset_run_id':DATA_RUN,'offset_coefficient':int(frequency)})
        for path in (Path(__file__),ROOT/'word_energy/text8_factorial_models.py',ROOT/'word_energy/text8_canonical_eval.py',ROOT/'word_energy/text8_lexical_models.py',ROOT/'word_energy/ising/exact_training.py',ROOT/'notes/TEXT8_FREQUENCY_SHUFFLE_PROTOCOL.md',ROOT/'scripts/run_text8_frequency_shuffle.py',ROOT/'scripts/train_text8_lexical_development.py',ROOT/'scripts/audit_text8_canonical_evaluator.py',ROOT/'word_energy/text8_development.py',ROOT/'tests/test_text8_factorial_models.py',ROOT/'tests/test_text8_canonical_eval.py'):
            mlflow.log_artifact(str(path),'source_snapshot')
        temp_root=ROOT/'.tmp'; temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as temporary:
            tmp=Path(temporary)
            details['zero_coupling_control']=zero_control(params,arm,offset,arrays,manifest['frequency_only_control'])
            details['checkpoint_hashes']['initial']=snapshot(params,'initial',tmp)
            details['metrics']['initial']=val_metrics(params,arm,score_fn,loss_fn,arrays)
            log_metrics('initial',details['metrics']['initial'])
            for epoch in range(3):
                total=0.; count_examples=0
                for i in range(0,len(arrays['train_pairs']),BATCH):
                    ix=arrays['orders'][epoch,i:i+BATCH]
                    params,state,value=step(params,state,arrays['shuffled_train_pairs'][ix],arrays['train_noise'][epoch,ix])
                    value=float(value)
                    if not np.isfinite(value): raise FloatingPointError('nonfinite training loss')
                    total+=value*len(ix); count_examples+=len(ix)
                stage=f'epoch{epoch+1}'; peak=peak_working_set()
                if peak>2*1024**3: raise MemoryError('process exceeded 2 GiB working set')
                details['checkpoint_hashes'][stage]=snapshot(params,stage,tmp)
                metrics=val_metrics(params,arm,score_fn,loss_fn,arrays)
                metrics['train_objective_mean']=total/count_examples; metrics['peak_working_set_mib']=peak/1024**2
                details['metrics'][stage]=metrics; log_metrics(stage,metrics)
                print(json.dumps({'run_id':run.info.run_id,'arm':arm,'seed':seed,'stage':stage,
                                  'auc':metrics['noise_pairwise_auc'],'mrr':metrics['noise_mrr_at_21'],'peak_mib':metrics['peak_working_set_mib']}),flush=True)
            details['elapsed_seconds']=time.monotonic()-start; details['run_complete']=True
            log_contract(mlflow,contract,details); mlflow.log_metric('resource.elapsed_seconds',details['elapsed_seconds'])
        print(json.dumps({'run_id':run.info.run_id,'arm':arm,'seed':seed,'status':'complete','elapsed_seconds':details['elapsed_seconds']}),flush=True)
