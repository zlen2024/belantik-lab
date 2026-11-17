"""Freeze samples and log a dataset preparation run; no model training."""
from pathlib import Path
import json
import sys
import tempfile
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from word_energy.text8_development import build
from word_energy.text8_lexical_models import pair_metrics
from word_energy.tracking import RunContract,log_contract,tracking_uri

if __name__=='__main__':
    start=time.monotonic(); mlflow.set_tracking_uri(tracking_uri())
    exp=mlflow.set_experiment('word-energy-text8-lexical-development')
    if MlflowClient().search_runs([exp.experiment_id],max_results=1000):
        raise RuntimeError('experiment already has runs; inspect instead of overwriting')
    arrays,manifest=build()
    val=arrays['validation_pairs']; noise=arrays['validation_noise']; counts=arrays['counts']
    score=.25*np.log(counts.astype(float))
    manifest['frequency_only_control']=pair_metrics(score[val[:,1]],score[noise],counts[val[:,0]])
    manifest['elapsed_seconds']=time.monotonic()-start
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='text8_lexical_data',
        energy_convention='no model training',variable_encoding='word IDs',beta_temperature_convention='not applicable',
        sampler_schedule='fixed pair/noise samples, not Gibbs sampling',seed=11301,data_hash=manifest['source']['content_sha256'],
        data_split='train bytes 10–11M, validation 16–16.5M; reserved 18–20M not evaluated',
        graph_summary='two-word occurrence pairs; window offsets ±1, ±2',limitations=manifest['limitations'])
    with mlflow.start_run(run_name='text8-data-frozen') as run:
        log_contract(mlflow,contract,manifest)
        for path in (Path(__file__),ROOT/'word_energy/text8_development.py',ROOT/'notes/TEXT8_LEXICAL_DEVELOPMENT_PROTOCOL.md'):
            mlflow.log_artifact(str(path),'source_snapshot')
        temp_root=ROOT/'.tmp'; temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as temp:
            path=Path(temp)/'samples.npz'; np.savez_compressed(path,**arrays)
            mlflow.log_artifact(str(path),'dataset')
        print(json.dumps({'run_id':run.info.run_id,'experiment_id':exp.experiment_id,'elapsed_seconds':manifest['elapsed_seconds'],
                          'train_stats':manifest['train_stats'],'validation_stats':manifest['validation_stats'],
                          'validation_exact_noise_collisions':manifest['validation_exact_noise_collisions'],
                          'train_unique_pairs':manifest['train_unique_pairs'],'validation_unique_pairs':manifest['validation_unique_pairs'],
                          'hashes':manifest['hashes']}))
