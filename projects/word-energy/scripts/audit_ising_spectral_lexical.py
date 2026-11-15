"""Read-only lexical400 AUC audit for nine completed initializer runs."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from word_energy.ising.exact_training import _free_energy_factory, _lexical_auc, prepare_data
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri

RUNS={'random':{101:'c9bf31a9746a44b3b65c699e81b6f0ee',103:'ef96b4fe6043436b8be8032be4e1469e',107:'0e8448a6b2464b05824c0da0c14e4ff4'},'spectral':{101:'52f0c53ff42043c9befdf6cf907748ac',103:'e4ce7df27d6243269c9fb73cb90badb1',107:'52643ad42d944b73961e900ae0eea112'},'permuted':{101:'be4a881aa08d46b6a808593187d597c2',103:'2ddccd344d814a47a31d7a691085d4c1',107:'72e3c0a9e7c54526a658917c54137e12'}}

if __name__=='__main__':
    data=prepare_data(); _,jnp,energy,_,_=_free_energy_factory(True); results={}
    for arm,runs in RUNS.items():
        results[arm]={}
        for seed,run_id in runs.items():
            path=ROOT/'mlruns'/'3'/run_id/'artifacts'/'config.json'
            payload=json.loads(path.read_text(encoding='utf-8'))
            if payload['seed']!=seed or payload['arm']!=arm: raise ValueError('run identity mismatch')
            raw={k:np.asarray(v,dtype=np.float64) for k,v in payload['snapshots']['stage1_lexical_400'].items()}
            diag=payload['fixed_lexical400_diagnosis']; auc=_lexical_auc(raw,energy,data,jnp)
            gain=(diag['family_summary']['F']['gain']+diag['family_summary']['R']['gain'])/2
            accuracy=diag['validation_macro_sense_accuracy']
            results[arm][str(seed)]={'source_run_id':run_id,'source_artifact_sha256':sha256_file(path),'lexical400_auc':auc,'lexical400_sense':accuracy,'lexical400_gain':gain,'pass':bool(accuracy>=10/12 and auc>=.75 and gain>=.03)}
    mlflow.set_tracking_uri(tracking_uri()); mlflow.set_experiment('word-energy-exact-training')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='ising_spectral_lexical400_audit',energy_convention='dimensionless beta E; normalized F2/F3',variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',sampler_schedule='read-only exact lexical scoring on saved snapshots',seed=0,data_hash=sha256_file(ROOT/'experiments/runs/synthetic-split-manifest.json'),data_split='training and validation only',graph_summary='two-spin tokens and lexical F2 scoring',limitations='Development toy; historical validation already used; no test or hardware.')
    with mlflow.start_run(run_name='ising-spectral-lexical400-auc-audit') as run:
        log_contract(mlflow,contract,{'source_run_ids':RUNS,'results':results})
        for arm,entries in results.items():
            for seed,values in entries.items():
                mlflow.log_metrics({f'{arm}.seed{seed}.auc':values['lexical400_auc'],f'{arm}.seed{seed}.sense':values['lexical400_sense'],f'{arm}.seed{seed}.gain':values['lexical400_gain'],f'{arm}.seed{seed}.pass':float(values['pass'])})
        print(json.dumps({'audit_run_id':run.info.run_id,'results':results}))
