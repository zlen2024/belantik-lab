"""Train a single preregistered corrected-F3 initializer ablation arm."""
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
from word_energy.ising.exact_training import ExactTrainingConfig, _free_energy_factory, _imports, _lexical_auc, prepare_data, train_smoke
from word_energy.ising.spectral_initializer import array_hash, spectral_biases
from word_energy.synthetic import vocabulary
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri


def random_initial(seed, vocab_size):
    jax,jnp,_=_imports()
    key=jax.random.key(seed)
    key,kb,kj=jax.random.split(key,3)
    return {'biases':np.asarray(jax.random.normal(kb,(vocab_size,2),dtype=jnp.float64)*.1), 'internal':np.asarray(jax.random.normal(kj,(vocab_size,),dtype=jnp.float64)*.1), 'interaction':np.asarray(jnp.full((2,),jnp.log(.8/(1.5-.8)),dtype=jnp.float64))}


@contextmanager
def persist_partial_on_error(seed, arm, snapshots):
    """Preserve captured snapshots on a Python exception, without hiding failure."""
    try:
        yield
    except Exception as error:
        try:
            contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='ising_train_cooccurrence_spectral_ablation_partial',energy_convention='beta E; F3 each spoke lambda/2, F2 each spoke lambda',variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',sampler_schedule='interrupted exact enumeration training; see captured snapshots',seed=seed,data_hash=sha256_file(ROOT/'experiments/runs/synthetic-split-manifest.json'),data_split='training and validation only',graph_summary='3 tokens, 2 spins each, star',limitations='Partial failed run; no test or physical hardware; abrupt process kill may prevent recovery.')
            log_contract(mlflow,contract,{'arm':arm,'seed':seed,'error_type':type(error).__name__,'error':str(error),'partial_snapshots':snapshots,'partial':True})
            mlflow.set_tag('partial_snapshots_captured','true')
        except Exception as logging_error:
            print(f'Could not persist partial snapshots: {type(logging_error).__name__}: {logging_error}',file=sys.stderr)
        raise


if __name__=='__main__':
    seed=int(sys.argv[1]); arm=sys.argv[2]
    if seed not in (101,103,107) or arm not in ('random','spectral','permuted'):
        raise ValueError('only frozen development seeds and arms are allowed')
    elapsed_start=time.monotonic(); data=prepare_data(); vocab=vocabulary()
    spectral=spectral_biases(data['train_windows'],len(vocab))
    initial=random_initial(seed,len(vocab)); random_copy={k:v.copy() for k,v in initial.items()}
    if arm!='random': initial['biases']=spectral['biases' if arm=='spectral' else 'permuted_biases'].copy()
    assert np.array_equal(initial['internal'],random_copy['internal']) and np.array_equal(initial['interaction'],random_copy['interaction'])
    snapshots={}
    def capture(prefix):
        def save(stage,step,raw):
            if not all(np.isfinite(v).all() for v in raw.values()): raise FloatingPointError(f'{prefix}:{stage}:{step}')
            if peak_working_set()>2*1024**3: raise MemoryError('2 GiB process peak working-set limit exceeded')
            if time.monotonic()-elapsed_start>1800: raise TimeoutError('30 minute per-process limit exceeded')
            snapshots[f'{prefix}_{stage}_{step}']={k:v.tolist() for k,v in raw.items()}
        return save
    mlflow.set_tracking_uri(tracking_uri()); mlflow.set_experiment('word-energy-exact-training')
    with mlflow.start_run(run_name=f'ising-spectral-init-{arm}-seed{seed}') as run, persist_partial_on_error(seed,arm,snapshots):
        mlflow.set_tags({'execution_tier':'classical','device_class':'cpu_gpu_simulator','hypothesis':'train-only cooccurrence bias helps separate contexts','test_status':'not evaluated; old holdout already inspected','arm':arm,'seed':str(seed),'normalize_spokes':'True'})
        for path in (Path(__file__).resolve(),ROOT/'word_energy/ising/spectral_initializer.py',ROOT/'word_energy/ising/exact_training.py'):
            mlflow.log_artifact(str(path),'source_snapshot')
        lexical_cfg=ExactTrainingConfig(seed=seed,lexical_updates=400,joint_updates=0,normalize_spokes=True)
        _,lex_metrics=train_smoke(lexical_cfg,snapshot_callback=capture('stage1'),initial_override=initial)
        fixed_lex={k:np.asarray(v,dtype=np.float64) for k,v in snapshots['stage1_lexical_400'].items()}
        joint_cfg=ExactTrainingConfig(seed=seed,lexical_updates=0,joint_updates=100,normalize_spokes=True)
        _,joint_metrics=train_smoke(joint_cfg,snapshot_callback=capture('stage2'),initial_override=fixed_lex)
        start_joint=snapshots['stage2_initial_0']
        if any(not np.array_equal(fixed_lex[k],np.asarray(start_joint[k])) for k in fixed_lex): raise ValueError('lexical400 handoff mismatch')
        fixed_joint={k:np.asarray(v,dtype=np.float64) for k,v in snapshots['stage2_joint_100'].items()}
        lex_diag=diagnose_raw(fixed_lex,seed,f'{arm}:fixed-lexical400',snapshot_hash(snapshots['stage1_lexical_400']),normalize_spokes=True)
        joint_diag=diagnose_raw(fixed_joint,seed,f'{arm}:fixed-joint100',snapshot_hash(snapshots['stage2_joint_100']),normalize_spokes=True)
        _,jnp,energy,_,_=_free_energy_factory(True)
        auc=_lexical_auc(fixed_joint,energy,data,jnp)
        score=joint_diag['validation_macro_sense_accuracy']
        gain=(joint_diag['family_summary']['F']['gain']+joint_diag['family_summary']['R']['gain'])/2
        passed=bool(score>=10/12 and auc>=.75 and gain>=.03)
        peak=peak_working_set(); elapsed=time.monotonic()-elapsed_start
        if peak>2*1024**3 or elapsed>1800: raise RuntimeError('resource budget exceeded before logging')
        def clean(metrics): return {k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in metrics.items()}
        initializer={'train_token_hash':array_hash(data['train_windows']),'vocabulary':list(vocab),'vocabulary_hash':snapshot_hash(list(vocab)),'count_matrix_hash':array_hash(spectral['counts']),'eigenvectors_hash':array_hash(spectral['eigenvectors']),'eigenvalues':spectral['eigenvalues'].tolist(),'eigengap_tolerance':spectral['eigengap_tolerance'],'bias_hash':array_hash(spectral['biases']),'permuted_bias_hash':array_hash(spectral['permuted_biases']),'row_permutation_seed':6061,'row_permutation':spectral['permutation'].tolist(),'arm_initial_bias_hash':array_hash(initial['biases'])}
        config={'seed':seed,'arm':arm,'lexical_config':lexical_cfg.__dict__,'joint_config':joint_cfg.__dict__,'initializer':initializer,'lexical_metrics':clean(lex_metrics),'joint_metrics':clean(joint_metrics),'fixed_lexical400_diagnosis':lex_diag,'fixed_joint100_diagnosis':joint_diag,'validation_lexical_auc':auc,'gain':gain,'passes_original_thresholds_on_development_validation':passed,'snapshots':snapshots,'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},'peak_process_working_set_bytes':peak,'elapsed_seconds':elapsed,'initializer_source_sha256':sha256_file(ROOT/'word_energy/ising/spectral_initializer.py'),'training_source_sha256':sha256_file(ROOT/'word_energy/ising/exact_training.py'),'runner_sha256':sha256_file(Path(__file__).resolve())}
        contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='ising_train_cooccurrence_spectral_ablation',energy_convention='beta E; F3 each spoke lambda/2, F2 each spoke lambda',variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',sampler_schedule='exact enumeration; fixed 400 lexical + 100 joint Adam updates',seed=seed,data_hash=sha256_file(ROOT/'experiments/runs/synthetic-split-manifest.json'),data_split='train-only initializer, training and previously inspected validation; no test evaluation',graph_summary='3 active tokens, 2 spins each, star',limitations='Development toy; initializer label-free but original negatives/corruptions use grammar oracle; three seeds; no physical hardware; resource checks at snapshots only.')
        log_contract(mlflow,contract,config)
        mlflow.log_params({'arm':arm,'seed':seed,'handoff':'fixed_lexical400','F3':'normalized_spokes','row_permutation_seed':6061})
        flat={'fixed100.sense_accuracy':score,'fixed100.context_gain':gain,'fixed100.lexical_auc':auc,'fixed100.anchor_tv':joint_diag['anchor_total_variation'],'fixed100.context_tv':joint_diag['context_posterior_total_variation'],'fixed100.pass_thresholds':float(passed),'lexical400.sense_accuracy':lex_diag['validation_macro_sense_accuracy'],'resource.peak_working_set_bytes':float(peak),'resource.elapsed_seconds':elapsed}
        for family,values in joint_diag['family_summary'].items():
            for key,value in values.items(): flat[f'fixed100.{family}.{key}']=value
        mlflow.log_metrics(flat)
        print(json.dumps({'run_id':run.info.run_id,'seed':seed,'arm':arm,'sense':score,'gain':gain,'auc':auc,'pass':passed,'anchor_tv':joint_diag['anchor_total_variation'],'peak_mib':peak/1024**2,'elapsed_seconds':elapsed}))
