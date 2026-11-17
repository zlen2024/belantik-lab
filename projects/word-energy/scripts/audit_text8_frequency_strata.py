"""Posthoc frequency-ratio breakdown of fixed validation comparisons."""
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
from word_energy.text8_lexical_models import ARMS,SEEDS,functions
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file

DATA='6cf76591c3fa49c686a3d81d7f225607'
CORRECTED='e14d3b8e72e1467597b3826575be896f'
NAMES=('lt_half','half_to_one','equal','one_to_two','gt_two')


def artifacts(client,run_id):
    run=client.get_run(run_id)
    if run.info.status!='FINISHED': raise ValueError('unfinished input run')
    uri=urlparse(run.info.artifact_uri)
    if uri.scheme!='file': raise ValueError('expected local artifacts')
    return Path(url2pathname(uri.path))


def groups(arrays):
    pairs=arrays['validation_pairs']; noise=arrays['validation_noise']; counts=arrays['counts']
    p=counts[pairs[:,1]][:,None]; n=counts[noise]
    distinct=noise!=pairs[:,1,None]
    masks=(2*p<n,(2*p>=n)&(p<n),p==n,(p>n)&(p<=2*n),p>2*n)
    result={key:mask&distinct for key,mask in zip(NAMES,masks)}
    if not np.array_equal(sum(x.astype(int) for x in result.values()),distinct.astype(int)):
        raise ValueError('frequency groups do not partition distinct comparisons')
    return result,distinct


def breakdown(p,n,masks):
    p=np.asarray(p)[:,None]; n=np.asarray(n)
    comparison=(p>n)+.5*(p==n)
    return {name:{'comparisons':int(mask.sum()),'auc':float(comparison[mask].mean()) if mask.any() else None}
            for name,mask in masks.items()}


def audit():
    client=MlflowClient(); arrays,manifest=get_data(DATA)
    corrected_root=artifacts(client,CORRECTED)
    correct=json.loads((corrected_root/'config.json').read_text(encoding='utf-8'))
    if correct['dataset_hashes']!=manifest['hashes']: raise ValueError('dataset mismatch')
    for relative in ('word_energy/text8_lexical_models.py','word_energy/ising/exact_training.py'):
        local=ROOT/relative
        if sha256_file(local)!=sha256_file(corrected_root/'source_snapshot'/local.name):
            raise ValueError(f'scorer source mismatch: {relative}')
    masks,distinct=groups(arrays); pairs=arrays['validation_pairs']; noise=arrays['validation_noise']
    base=.25*np.log(arrays['counts'].astype(float))
    frequency=breakdown(base[pairs[:,1]],base[noise],masks)
    result={}
    for arm in ARMS:
        score_fn,_,_,_=functions(arm)
        for seed in SEEDS:
            original=correct['runs'][f'{arm}:{seed}']; root=artifacts(client,original['run_id'])
            with np.load(root/'checkpoints'/'epoch3.npz',allow_pickle=False) as checkpoint:
                params={k:checkpoint[k] for k in checkpoint.files}
            if {k:sha_array(v) for k,v in params.items()}!=original['stages']['epoch3']['checkpoint_hashes']:
                raise ValueError('checkpoint mismatch')
            pos=[]; neg=[]
            for i in range(0,len(pairs),500):
                p,n=score_fn(params,pairs[i:i+500],noise[i:i+500])
                pos.append(np.asarray(p)); neg.append(np.asarray(n))
            p=np.concatenate(pos); n=np.concatenate(neg); strata=breakdown(p,n,masks)
            weighted=sum(x['auc']*x['comparisons'] for x in strata.values() if x['auc'] is not None)/distinct.sum()
            direct=float(((p[:,None]>n)+.5*(p[:,None]==n))[distinct].mean())
            if not np.isclose(weighted,direct,atol=1e-12,rtol=0): raise ValueError('strata recombination mismatch')
            result[f'{arm}:{seed}']={'run_id':original['run_id'],'groups':strata,'distinct_id_auc':direct}
    paired={name:[result[f'ising-ranking:{s}']['groups'][name]['auc']-result[f'ising-shuffled:{s}']['groups'][name]['auc']
                  if result[f'ising-ranking:{s}']['groups'][name]['auc'] is not None else None for s in SEEDS] for name in NAMES}
    paired_means={name:float(np.mean(values)) if all(value is not None for value in values) else None for name,values in paired.items()}
    return {'dataset_run_id':DATA,'corrected_audit_run_id':CORRECTED,'dataset_hashes':manifest['hashes'],
            'source_sha256':manifest['source']['content_sha256'],'distinct_comparisons':int(distinct.sum()),
            'same_id_excluded':int((~distinct).sum()),'frequency_only':frequency,'runs':result,'paired_ising_minus_shuffle':paired,'paired_mean_ising_minus_shuffle':paired_means,
            'limitations':'posthoc groups on same text8 validation, shared corpus across seeds and repeated occurrences; frequency strata descriptive not causal; no test/sense/hardware'}


if __name__=='__main__':
    start=time.monotonic(); mlflow.set_tracking_uri(tracking_uri()); result=audit(); result['elapsed_seconds']=time.monotonic()-start
    mlflow.set_experiment('word-energy-text8-lexical-development')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='text8_frequency_strata_posthoc_audit',
        energy_convention='dimensionless beta E or vector dot score',variable_encoding='spins {-1,+1} or 2D vectors',
        beta_temperature_convention='beta=1 for Ising',sampler_schedule='saved checkpoint evaluation only',seed=0,
        data_hash=result['source_sha256'],data_split='same text8 development validation; no test',
        graph_summary='2-token F2 or 2D vector tables',limitations=result['limitations'])
    with mlflow.start_run(run_name='text8-frequency-strata-posthoc') as run:
        log_contract(mlflow,contract,result)
        for path in (Path(__file__).resolve(),ROOT/'notes/TEXT8_FREQUENCY_STRATA_POSTHOC_PROTOCOL.md',
                     ROOT/'word_energy/text8_lexical_models.py',ROOT/'word_energy/ising/exact_training.py'):
            mlflow.log_artifact(str(path),'source_snapshot')
        print(json.dumps({'run_id':run.info.run_id,'frequency_only':result['frequency_only'],
                          'paired_ising_minus_shuffle':result['paired_ising_minus_shuffle'],'paired_mean_ising_minus_shuffle':result['paired_mean_ising_minus_shuffle'],
                          'elapsed_seconds':result['elapsed_seconds']}))
