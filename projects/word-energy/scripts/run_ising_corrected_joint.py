"""Validation-only F3 normalization sensitivity and corrected joint training."""
import ctypes
from ctypes import wintypes
import hashlib
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
from word_energy.ising.exact_training import ExactTrainingConfig, _free_energy_factory, _lexical_auc, prepare_data, train_smoke
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri


class ProcessMemoryCounters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]


def peak_working_set():
    if os.name != 'nt': raise RuntimeError('peak memory measurement is required on Windows')
    values=ProcessMemoryCounters(); values.cb=ctypes.sizeof(values)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True); kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(ProcessMemoryCounters),wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(values),values.cb): raise ctypes.WinError(ctypes.get_last_error())
    return int(values.PeakWorkingSetSize)


def snapshot_hash(snapshot):
    return hashlib.sha256(json.dumps(snapshot,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()


if __name__=='__main__':
    seed=int(sys.argv[1])
    if seed not in (17,29,43): raise ValueError('only preregistered development seeds')
    source=ROOT/'experiments'/'runs'/f'ising-attribution-original-seed{seed}.json'
    replay=json.loads(source.read_text(encoding='utf-8'))
    if replay.get('reproduction_pass') is not True or replay['selected_metrics']['selection.lexical_step']!=400: raise ValueError('lexical replay provenance invalid')
    initial={k:np.asarray(v,dtype=np.float64) for k,v in replay['snapshots']['lexical_selected_400'].items()}
    baseline_hash=sha256_file(source)
    lexical_hash=snapshot_hash(replay['snapshots']['lexical_selected_400'])
    old=diagnose_raw(initial,seed,'lexical400-legacy',lexical_hash,normalize_spokes=False)
    new=diagnose_raw(initial,seed,'lexical400-corrected',lexical_hash,normalize_spokes=True)
    snapshots={}; start=time.monotonic()
    def capture(stage,step,values):
        if not all(np.isfinite(value).all() for value in values.values()): raise FloatingPointError(f'nonfinite {stage} {step}')
        if peak_working_set()>2*1024**3: raise MemoryError('process peak working set exceeded 2 GiB')
        if time.monotonic()-start>1800: raise TimeoutError('30-minute limit exceeded')
        snapshots[f'{stage}_{step}']={k:value.tolist() for k,value in values.items()}
    config=ExactTrainingConfig(seed=seed,lexical_updates=0,joint_updates=100,normalize_spokes=True)
    _,metrics=train_smoke(config,snapshot_callback=capture,initial_override=initial)
    elapsed=time.monotonic()-start
    selected_step=metrics['selection.joint_step']
    selected={k:np.asarray(v,dtype=np.float64) for k,v in snapshots[f'joint_selected_{selected_step}'].items()}
    fixed={k:np.asarray(v,dtype=np.float64) for k,v in snapshots['joint_100'].items()}
    selected_diag=diagnose_raw(selected,seed,'corrected-selected',snapshot_hash(snapshots[f'joint_selected_{selected_step}']),normalize_spokes=True)
    fixed_diag=diagnose_raw(fixed,seed,'corrected-step100',snapshot_hash(snapshots['joint_100']),normalize_spokes=True)
    _,jnp,energy,_,_=_free_energy_factory(True); data=prepare_data()
    auc=_lexical_auc(selected,energy,data,jnp)
    metrics={k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in metrics.items()}
    payload={'seed':seed,'config':config.__dict__,'lexical_source_sha256':baseline_hash,'runner_sha256':sha256_file(Path(__file__).resolve()),'training_source_sha256':sha256_file(ROOT/'word_energy/ising/exact_training.py'),'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},'peak_process_working_set_bytes':peak_working_set(),'elapsed_training_seconds':elapsed,'same_checkpoint_legacy':old,'same_checkpoint_corrected':new,'selected_corrected':selected_diag,'fixed100_corrected':fixed_diag,'selected_validation_lexical_auc':auc,'training_metrics':metrics,'snapshots':snapshots}
    if payload['peak_process_working_set_bytes']>2*1024**3: raise MemoryError('peak exceeded 2 GiB')
    mlflow.set_tracking_uri(tracking_uri()); mlflow.set_experiment('word-energy-exact-training')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='ising_corrected_f3_joint_baseline',energy_convention='beta E; F3 each spoke lambda/2, F2 each spoke lambda',variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',sampler_schedule='exact enumeration, 100 joint optimizer updates, lexical checkpoint reused',seed=seed,data_hash=sha256_file(ROOT/'experiments/runs/synthetic-split-manifest.json'),data_split='training and validation only',graph_summary='3 tokens, 2 spins each, star',limitations='Exploratory corrected baseline on previously inspected synthetic seeds and validation; original test was already opened; no hardware.')
    with mlflow.start_run(run_name=f'ising-corrected-f3-validation-seed{seed}') as run:
        log_contract(mlflow,contract,payload)
        mlflow.log_params({'seed':seed,'F3_spoke_normalization':'1/(L-1)','joint_updates':100,'lexical_source':'verified replay selected step400'})
        flat={}
        for name,diag in [('lexical_legacy',old),('lexical_corrected',new),('joint_selected',selected_diag),('joint_100',fixed_diag)]:
            for key in ('anchor_total_variation','context_posterior_total_variation','validation_macro_sense_accuracy','validation_joint_ranking_loss'): flat[f'{name}.{key}']=diag[key]
            for family,values in diag['family_summary'].items():
                for key,value in values.items(): flat[f'{name}.{family}.{key}']=value
        flat.update({'selected.validation_lexical_auc':auc,'resource.peak_working_set_bytes':float(payload['peak_process_working_set_bytes']),'resource.elapsed_training_seconds':elapsed})
        mlflow.log_metrics(flat)
        print(json.dumps({'run_id':run.info.run_id,'seed':seed,'selected_step':selected_step,'selected_sense':selected_diag['validation_macro_sense_accuracy'],'fixed100_sense':fixed_diag['validation_macro_sense_accuracy'],'selected_gain':(selected_diag['family_summary']['F']['gain']+selected_diag['family_summary']['R']['gain'])/2,'auc':auc,'peak_mib':payload['peak_process_working_set_bytes']/1024**2,'elapsed_seconds':elapsed}))
