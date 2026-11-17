"""Repair exact same-ID ties in saved checkpoint metrics; never retrains."""
from pathlib import Path
import json
import sys
import time
import numpy as np
from urllib.parse import urlparse
from urllib.request import url2pathname

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.train_text8_lexical_development import get_data
from word_energy.text8_development import sha_array
from word_energy.text8_lexical_models import ARMS,SEEDS,functions,pair_metrics
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file

DATA_RUN='6cf76591c3fa49c686a3d81d7f225607'
RAW_SUMMARY='790289d209054c48882b934809e60049'


def repair_duplicate_scores(positive,negatives,positive_ids,noise_ids):
    same=np.asarray(noise_ids)==np.asarray(positive_ids)[:,None]
    difference=np.abs(np.asarray(negatives)-np.asarray(positive)[:,None])
    error=float(difference[same].max()) if np.any(same) else 0.
    if error>1e-12: raise ValueError('duplicate candidate scores differ beyond rounding')
    return np.where(same,np.asarray(positive)[:,None],negatives),int(np.sum(same & (difference>0))),error


def run_dir(client,run_id):
    run=client.get_run(run_id)
    if run.info.status!='FINISHED': raise ValueError(f'unfinished source {run_id}')
    uri=urlparse(run.info.artifact_uri)
    if uri.scheme!='file': raise ValueError('expected local store')
    return Path(url2pathname(uri.path))


def audit():
    client=MlflowClient(); arrays,manifest=get_data(DATA_RUN)
    summary=json.loads((run_dir(client,RAW_SUMMARY)/'config.json').read_text(encoding='utf-8'))
    if summary['dataset_hashes']!=manifest['hashes'] or len(summary['runs'])!=12: raise ValueError('summary/data mismatch')
    pairs=arrays['validation_pairs']; noise=arrays['validation_noise']; same=noise==pairs[:,1,None]
    if int(same.sum())!=manifest['validation_exact_noise_collisions']: raise ValueError('collision count mismatch')
    result={}; total_false=0; max_error=0.
    for arm in ARMS:
        score_fn,_,_,_=functions(arm)
        for seed in SEEDS:
            row=summary['runs'][f'{arm}:{seed}']; root=run_dir(client,row['run_id'])
            cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
            if cfg['dataset_hashes']!=manifest['hashes']: raise ValueError('run dataset mismatch')
            for relative in ('word_energy/text8_lexical_models.py','word_energy/ising/exact_training.py',
                             'word_energy/text8_development.py','scripts/train_text8_lexical_development.py'):
                local=ROOT/relative; historical=root/'source_snapshot'/local.name
                if sha256_file(local)!=sha256_file(historical): raise ValueError(f'source mismatch {relative}')
            stages={}
            for stage in ('initial','epoch1','epoch2','epoch3'):
                with np.load(root/'checkpoints'/f'{stage}.npz',allow_pickle=False) as checkpoint:
                    params={k:checkpoint[k] for k in checkpoint.files}
                if {k:sha_array(v) for k,v in params.items()}!=cfg['checkpoint_hashes'][stage]: raise ValueError('checkpoint hash mismatch')
                positive=[]; negatives=[]
                for start in range(0,len(pairs),500):
                    p,n=score_fn(params,pairs[start:start+500],noise[start:start+500])
                    positive.append(np.asarray(p)); negatives.append(np.asarray(n))
                positive=np.concatenate(positive); negatives=np.concatenate(negatives)
                fixed,false,error=repair_duplicate_scores(positive,negatives,pairs[:,1],noise)
                total_false+=false; max_error=max(max_error,error)
                raw=pair_metrics(positive,negatives,arrays['counts'][pairs[:,0]])
                for field,value in raw.items():
                    if not np.allclose(value,cfg['metrics'][stage][field],atol=1e-12,rtol=0):
                        raise ValueError(f'raw replay mismatch {arm}/{seed}/{stage}/{field}')
                corrected=pair_metrics(positive,fixed,arrays['counts'][pairs[:,0]])
                stages[stage]={'raw':raw,'corrected':corrected,'same_id_false_nonties':false,'same_id_max_abs_error':error,
                               'checkpoint_hashes':cfg['checkpoint_hashes'][stage]}
            result[f'{arm}:{seed}']={'run_id':row['run_id'],'source_config_sha256':sha256_file(root/'config.json'),'stages':stages}
    aggregate={}
    for arm in ARMS:
        auc=np.array([result[f'{arm}:{s}']['stages']['epoch3']['corrected']['noise_pairwise_auc'] for s in SEEDS])
        mrr=np.array([result[f'{arm}:{s}']['stages']['epoch3']['corrected']['noise_mrr_at_21'] for s in SEEDS])
        aggregate[arm]={'noise_auc_mean':float(auc.mean()),'noise_auc_seed_sd':float(auc.std(ddof=1)),
                        'noise_mrr_mean':float(mrr.mean()),'noise_mrr_seed_sd':float(mrr.std(ddof=1)),
                        'seed_auc':auc.tolist(),'seed_mrr':mrr.tolist()}
    return {'data_run_id':DATA_RUN,'raw_summary_run_id':RAW_SUMMARY,'dataset_hashes':manifest['hashes'],
            'source_sha256':manifest['source']['content_sha256'],'identity_collisions_per_stage':int(same.sum()),
            'total_false_nonties':total_false,'maximum_identity_score_error':max_error,
            'frequency_only_control':manifest['frequency_only_control'],'runs':result,'aggregate':aggregate,
            'limitations':'posthoc exact-ID tie repair on same development validation; training and checkpoints unchanged, no test/word senses/hardware; repeated candidates and shared corpus remain.'}


if __name__=='__main__':
    start=time.monotonic(); mlflow.set_tracking_uri(tracking_uri()); result=audit(); result['elapsed_seconds']=time.monotonic()-start
    mlflow.set_experiment('word-energy-text8-lexical-development')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='text8_lexical_identity_tie_metric_audit',
        energy_convention='dimensionless Ising beta E or vector score',variable_encoding='spins {-1,+1} or 2D vectors',
        beta_temperature_convention='beta=1 for Ising',sampler_schedule='saved checkpoint evaluation only; no training',seed=0,
        data_hash=result['source_sha256'],data_split='same development validation; reserved test not evaluated',
        graph_summary='two-token two-spin Ising F2 or vector tables',limitations=result['limitations'])
    with mlflow.start_run(run_name='text8-identity-tie-metric-audit') as run:
        log_contract(mlflow,contract,result)
        for path in (Path(__file__).resolve(),ROOT/'notes/TEXT8_IDENTICAL_CANDIDATE_METRIC_REPAIR.md',
                     ROOT/'word_energy/text8_lexical_models.py',ROOT/'word_energy/ising/exact_training.py',
                     ROOT/'word_energy/text8_development.py',ROOT/'scripts/train_text8_lexical_development.py'):
            mlflow.log_artifact(str(path),'source_snapshot')
        mlflow.log_metrics({f"{arm}.corrected_auc":v['noise_auc_mean'] for arm,v in result['aggregate'].items()})
        mlflow.log_metric('false_nonties_across_48_stages',result['total_false_nonties'])
        print(json.dumps({'audit_run_id':run.info.run_id,'elapsed_seconds':result['elapsed_seconds'],
                          'total_false_nonties':result['total_false_nonties'],'maximum_identity_score_error':result['maximum_identity_score_error'],
                          'frequency_control':result['frequency_only_control']['noise_pairwise_auc'],'aggregate':result['aggregate']}))
