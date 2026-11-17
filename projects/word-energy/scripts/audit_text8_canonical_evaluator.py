"""Validate new canonical evaluator against historical corrected checkpoints."""
from pathlib import Path
import json
import sys
import time
from urllib.parse import urlparse
from urllib.request import url2pathname
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.train_text8_lexical_development import get_data
from word_energy.ising.exact_training import _imports
from word_energy.text8_canonical_eval import evaluate,stable_ising_score_adapter,stable_tied_ising_scores
from word_energy.text8_development import sha_array
from word_energy.text8_lexical_models import ARMS,SEEDS,functions,pair_metrics
from word_energy.tracking import RunContract,log_contract,sha256_file,tracking_uri

DATA='6cf76591c3fa49c686a3d81d7f225607'
CORRECTED='e14d3b8e72e1467597b3826575be896f'


def artifact_dir(client,rid):
    run=client.get_run(rid)
    if run.info.status!='FINISHED': raise ValueError('unfinished source run')
    uri=urlparse(run.info.artifact_uri)
    if uri.scheme!='file': raise ValueError('expected local artifact store')
    return Path(url2pathname(uri.path))


def audit():
    client=MlflowClient(); arrays,manifest=get_data(DATA); jax,jnp,_=_imports()
    correction_root=artifact_dir(client,CORRECTED)
    correction=json.loads((correction_root/'config.json').read_text(encoding='utf-8'))
    if correction['dataset_hashes']!=manifest['hashes']: raise ValueError('data hash mismatch')
    for relative in ('word_energy/text8_lexical_models.py','word_energy/ising/exact_training.py'):
        local=ROOT/relative
        if sha256_file(local)!=sha256_file(correction_root/'source_snapshot'/local.name): raise ValueError('historical scorer mismatch')
    pairs=arrays['validation_pairs']; noise=arrays['validation_noise']; counts=arrays['counts'][pairs[:,0]]
    result={}; max_error=0.
    for arm in ARMS:
        score_fn=jax.jit(stable_ising_score_adapter) if arm.startswith('ising') else functions(arm)[0]
        for seed in SEEDS:
            previous=correction['runs'][f'{arm}:{seed}']; root=artifact_dir(client,previous['run_id'])
            with np.load(root/'checkpoints'/'epoch3.npz',allow_pickle=False) as archive:
                params={key:archive[key] for key in archive.files}
            if {k:sha_array(v) for k,v in params.items()}!=previous['stages']['epoch3']['checkpoint_hashes']:
                raise ValueError('checkpoint hash mismatch')
            metrics=evaluate(params,score_fn,pairs,noise,counts,symmetric=arm.startswith('ising'))
            old=previous['stages']['epoch3']['corrected']
            errors={key:abs(metrics[key]-old[key]) for key in ('noise_pairwise_auc','noise_mrr_at_21')}
            max_error=max(max_error,*errors.values())
            result[f'{arm}:{seed}']={'run_id':previous['run_id'],'canonical':metrics,'previous_corrected':old,'absolute_differences':errors}
            if arm.startswith('ising'):
                sample=jnp.asarray(pairs[:64])
                zero=np.asarray(stable_tied_ising_scores(params,sample,0.0))
                if not np.array_equal(zero,np.zeros(len(sample))): raise ValueError('stable zero coupling not exactly zero')
                control=evaluate(params,score_fn,pairs,noise,counts,symmetric=True,zero_coupling=True)
                expected=sum(1/i for i in range(1,22))/21
                if control['noise_pairwise_auc']!=.5 or not np.isclose(control['noise_mrr_at_21'],expected,atol=1e-15,rtol=0):
                    raise ValueError('zero coupling tie metrics invalid')
                result[f'{arm}:{seed}']['zero_coupling']=control
    return {'dataset_run_id':DATA,'previous_corrected_run_id':CORRECTED,'source_sha256':manifest['source']['content_sha256'],
            'dataset_hashes':manifest['hashes'],'max_metric_difference':max_error,'runs':result,
            'limitations':'posthoc evaluation of existing text8 development checkpoints; no new training/test/THRML/hardware'}


if __name__=='__main__':
    start=time.monotonic(); mlflow.set_tracking_uri(tracking_uri()); result=audit(); result['elapsed_seconds']=time.monotonic()-start
    mlflow.set_experiment('word-energy-text8-lexical-development')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='text8_canonical_evaluator_audit',
        energy_convention='normalized Ising F2 log overlap or vector dot',variable_encoding='spins {-1,+1} or 2D vectors',
        beta_temperature_convention='beta=1 for Ising',sampler_schedule='canonical unique-pair checkpoint evaluation; no training',seed=0,
        data_hash=result['source_sha256'],data_split='existing development validation only; reserved test untouched',
        graph_summary='two-token four-spin F2 or vector pair',limitations=result['limitations'])
    with mlflow.start_run(run_name='text8-canonical-evaluator-audit') as run:
        log_contract(mlflow,contract,result)
        for path in (Path(__file__).resolve(),ROOT/'word_energy/text8_canonical_eval.py',ROOT/'notes/TEXT8_CANONICAL_EVALUATOR_PROTOCOL.md'):
            mlflow.log_artifact(str(path),'source_snapshot')
        mlflow.log_metric('max_difference_from_previous_corrected',result['max_metric_difference'])
        print(json.dumps({'run_id':run.info.run_id,'elapsed_seconds':result['elapsed_seconds'],
                          'max_metric_difference':result['max_metric_difference'],
                          'differences':{key:value['absolute_differences'] for key,value in result['runs'].items()}}))
