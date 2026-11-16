"""One frozen fresh-synthetic replication arm; no test scoring."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from scripts.diagnose_ising_validation import diagnose_raw
from scripts.run_ising_corrected_joint import peak_working_set, snapshot_hash
from scripts.run_ising_spectral_ablation import random_initial
from word_energy.fresh_synthetic import REPLICATES, make_fresh_corpus
from word_energy.ising.exact_training import ExactTrainingConfig, _free_energy_factory, _lexical_auc, prepare_data, train_smoke
from word_energy.ising.spectral_initializer import array_hash, spectral_biases
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri

SEEDS=(251,257,263,269)
ARMS=('random','spectral','permuted')


@contextmanager
def log_partial(seed,replicate,arm,permutation_seed,data_hash,snapshots):
    try:
        yield
    except Exception as error:
        try:
            contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='ising_fresh_synthetic_partial',energy_convention='beta E; corrected F3 lambda/2',variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',sampler_schedule='interrupted exact training; see snapshots',seed=seed,data_hash=data_hash,data_split='fresh corpus training and validation only',graph_summary='3 token star, 2 spins each',limitations='Failed/incomplete run; no test or hardware; partial recovery on Python exception only.')
            log_contract(mlflow,contract,{'replicate':replicate,'seed':seed,'arm':arm,'permutation_seed':permutation_seed,'error_type':type(error).__name__,'error':str(error),'snapshots':snapshots})
            mlflow.set_tag('partial_snapshots_captured','true')
        except Exception as logging_error:
            print(f'Failed to log partial state: {logging_error}',file=sys.stderr)
        raise


if __name__=='__main__':
    replicate=227; seed=int(sys.argv[1]); permutation_seed=int(sys.argv[2]); arm='permuted'
    if seed not in SEEDS or permutation_seed not in (7011,7013,7019,7027,7039,7043,7057,7069): raise ValueError('run outside frozen protocol')
    start=time.monotonic(); corpus=make_fresh_corpus(replicate); data=prepare_data(corpus=corpus)
    vocab=corpus['vocabulary']; snapshots={}
    mlflow.set_tracking_uri(tracking_uri()); mlflow.set_experiment('word-energy-corpus227-permutation-panel-v1')
    with mlflow.start_run(run_name=f'corpus227-perm{permutation_seed}-modelseed{seed}') as run, log_partial(seed,replicate,arm,permutation_seed,data['protocol_sha256'],snapshots):
        mlflow.set_tags({'execution_tier':'classical','device_class':'cpu_gpu_simulator','replicate':str(replicate),'seed':str(seed),'arm':arm,'permutation_seed':str(permutation_seed),'test_status':'reserved test not evaluated','protocol_sha256':data['protocol_sha256']})
        for path in (Path(__file__).resolve(),ROOT/'word_energy/fresh_synthetic.py',ROOT/'word_energy/ising/spectral_initializer.py',ROOT/'word_energy/ising/exact_training.py',ROOT/'scripts/diagnose_ising_validation.py'):
            mlflow.log_artifact(str(path),'source_snapshot')
        mlflow.log_artifact(str(ROOT/'notes/ISING_CORPUS227_PERMUTATION_PANEL_PROTOCOL.md'), 'protocol')
        initial=random_initial(seed,len(vocab)); baseline={k:v.copy() for k,v in initial.items()}
        spectral_meta=None
        if arm!='random':
            spectral=spectral_biases(data['train_windows'],len(vocab))
            permutation=np.random.default_rng(permutation_seed).permutation(len(vocab))
            initial['biases']=spectral['biases'][permutation].copy()
            spectral_meta={'count_hash':array_hash(spectral['counts']),'eigenvectors_hash':array_hash(spectral['eigenvectors']),'bias_hash':array_hash(spectral['biases']),'permuted_bias_hash':array_hash(initial['biases']),'eigenvalues':spectral['eigenvalues'].tolist(),'permutation':permutation.tolist(),'permutation_seed':permutation_seed,'eigengap_tolerance':spectral['eigengap_tolerance']}
        if not np.array_equal(initial['internal'],baseline['internal']) or not np.array_equal(initial['interaction'],baseline['interaction']): raise ValueError('paired initialization differs outside bias')
        def capture(prefix):
            def save(stage,step,raw):
                if not all(np.isfinite(v).all() for v in raw.values()): raise FloatingPointError(f'{prefix}:{stage}:{step}')
                if peak_working_set()>2*1024**3: raise MemoryError('2 GiB process working set exceeded')
                if time.monotonic()-start>3600: raise TimeoutError('batch time upper bound exceeded')
                snapshots[f'{prefix}_{stage}_{step}']={k:v.tolist() for k,v in raw.items()}
            return save
        lexical_cfg=ExactTrainingConfig(seed=seed,lexical_updates=400,joint_updates=0,normalize_spokes=True)
        _,lex_metrics=train_smoke(lexical_cfg,snapshot_callback=capture('stage1'),initial_override=initial,data_override=data)
        fixed_lex={k:np.asarray(v,dtype=np.float64) for k,v in snapshots['stage1_lexical_400'].items()}
        joint_cfg=ExactTrainingConfig(seed=seed,lexical_updates=0,joint_updates=100,normalize_spokes=True)
        _,joint_metrics=train_smoke(joint_cfg,snapshot_callback=capture('stage2'),initial_override=fixed_lex,data_override=data)
        if any(not np.array_equal(fixed_lex[k],np.asarray(snapshots['stage2_initial_0'][k])) for k in fixed_lex): raise ValueError('fixed lexical400 handoff mismatch')
        fixed_joint={k:np.asarray(v,dtype=np.float64) for k,v in snapshots['stage2_joint_100'].items()}
        lex_diag=diagnose_raw(fixed_lex,seed,'fresh-lexical400',snapshot_hash(snapshots['stage1_lexical_400']),normalize_spokes=True,data_override=data)
        joint_diag=diagnose_raw(fixed_joint,seed,'fresh-joint100',snapshot_hash(snapshots['stage2_joint_100']),normalize_spokes=True,data_override=data)
        _,jnp,energy,_,_=_free_energy_factory(True)
        lex_auc=_lexical_auc(fixed_lex,energy,data,jnp); joint_auc=_lexical_auc(fixed_joint,energy,data,jnp)
        def summary(diag,auc):
            sense=diag['validation_macro_sense_accuracy']; gain=(diag['family_summary']['F']['gain']+diag['family_summary']['R']['gain'])/2
            return {'sense':sense,'auc':auc,'gain':gain,'anchor_tv':diag['anchor_total_variation'],'context_tv':diag['context_posterior_total_variation'],'pass':bool(sense>=10/12 and auc>=.75 and gain>=.03)}
        lexical=summary(lex_diag,lex_auc); joint=summary(joint_diag,joint_auc)
        peak=peak_working_set(); elapsed=time.monotonic()-start
        if peak>2*1024**3 or elapsed>3600: raise RuntimeError('resource limit exceeded')
        def clean(m): return {k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in m.items()}
        payload={'replicate':replicate,'seed':seed,'arm':arm,'permutation_seed':permutation_seed,'protocol_sha256':data['protocol_sha256'],'schedule_sha256':data['schedule_sha256'],'train_token_hash':array_hash(data['train_windows']),'vocabulary':list(vocab),'pools':corpus['pools'],'pair_index_splits':{f:{k:v for k,v in splits.items() if k!='test'} for f,splits in corpus['pair_index_splits'].items()},'initial_hashes':{k:array_hash(v) for k,v in initial.items()},'baseline_initial_hashes':{k:array_hash(v) for k,v in baseline.items()},'initializer':spectral_meta,'lexical_config':lexical_cfg.__dict__,'joint_config':joint_cfg.__dict__,'lexical_metrics':clean(lex_metrics),'joint_metrics':clean(joint_metrics),'fixed_lexical400_diagnosis':lex_diag,'fixed_joint100_diagnosis':joint_diag,'fixed_lexical400':lexical,'fixed_joint100':joint,'snapshots':snapshots,'peak_process_working_set_bytes':peak,'elapsed_seconds':elapsed,'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')}}
        contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='ising_corpus227_permutation_panel_v1',energy_convention='beta E; F3 lambda/2 each spoke, F2 lambda',variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',sampler_schedule='exact enumeration; lexical400 + joint100 optimizer updates',seed=seed,data_hash=data['protocol_sha256'],data_split='posthoc selected corpus 227; train and validation only',graph_summary='3 tokens, 2 spins, star',limitations='Exploratory selected-corpus follow-up; synthetic oracle-assisted negatives/corruptions, family labels for sense map; no test/hardware; resource checks periodic.')
        log_contract(mlflow,contract,payload)
        mlflow.log_params({'replicate':replicate,'seed':seed,'arm':arm,'handoff':'fixed_lexical400','F3':'normalized_spokes','permutation_seed':permutation_seed})
        flat={}
        for name,results in (('lexical400',lexical),('joint100',joint)):
            for key,value in results.items(): flat[f'{name}.{key}']=float(value)
        flat.update({'resource.peak_working_set_bytes':float(peak),'resource.elapsed_seconds':elapsed})
        mlflow.log_metrics(flat)
        print(json.dumps({'run_id':run.info.run_id,'replicate':replicate,'seed':seed,'arm':arm,'permutation_seed':permutation_seed,'lexical400':lexical,'joint100':joint,'peak_mib':peak/1024**2,'elapsed_seconds':elapsed}))
