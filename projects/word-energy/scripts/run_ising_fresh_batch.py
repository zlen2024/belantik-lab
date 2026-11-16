"""Run the frozen 72-arm fresh synthetic panel under one wall-time budget."""
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from word_energy.fresh_synthetic import REPLICATES, make_fresh_corpus
from word_energy.ising.exact_training import prepare_data
from word_energy.ising.spectral_initializer import spectral_biases
from word_energy.tracking import RunContract, log_contract, tracking_uri
from scripts.run_ising_fresh_synthetic import ARMS, SEEDS


def record_spectrum_failure(rep,seed,arm,data_hash,reason):
    with mlflow.start_run(run_name=f'fresh-v2-corpus{rep}-{arm}-seed{seed}'):
        mlflow.set_tags({'execution_tier':'classical','device_class':'cpu_gpu_simulator','replicate':str(rep),'seed':str(seed),'arm':arm,'protocol_sha256':data_hash,'status_reason':'spectral_eigengap_failure','failure_detail':reason,'test_status':'not evaluated'})
        contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='ising_fresh_spectral_preflight_failure',energy_convention='beta E; corrected F3 lambda/2',variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',sampler_schedule='no training; spectral preflight failed',seed=seed,data_hash=data_hash,data_split=f'fresh corpus {rep} training input only',graph_summary='3 tokens, 2 spins each, star',limitations='Recorded method failure; no redraw, test, training or hardware.')
        log_contract(mlflow,contract,{'replicate':rep,'seed':seed,'arm':arm,'reason':reason})
        for path in (Path(__file__).resolve(),ROOT/'word_energy/fresh_synthetic.py',ROOT/'word_energy/ising/spectral_initializer.py'):
            mlflow.log_artifact(str(path),'source_snapshot')
        mlflow.log_metric('joint100.pass',0.0)
        mlflow.end_run(status='FAILED')


if __name__=='__main__':
    mlflow.set_tracking_uri(tracking_uri())
    experiment=mlflow.set_experiment('word-energy-fresh-synthetic-v2')
    client=MlflowClient()
    existing=client.search_runs([experiment.experiment_id],max_results=1000)
    if existing:
        raise RuntimeError('fresh benchmark experiment already has runs; inspect existing state before rerunning to avoid duplicates')
    env=os.environ.copy(); env.update({'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MLFLOW_DISABLE_AGENT_HINT':'1'})
    start=time.monotonic(); deadline=start+3600; total=0
    for rep in REPLICATES:
        data=prepare_data(corpus=make_fresh_corpus(rep))
        spectrum_error=None
        try:
            spectral_biases(data['train_windows'],len(data['ids']))
        except ValueError as exc:
            spectrum_error=str(exc)
            print(f'corpus={rep} spectral preflight failure: {spectrum_error}',flush=True)
        for seed in SEEDS:
            for arm in ARMS:
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    print(f'BATCH TIME LIMIT: completed={total}, expected=72; incomplete',flush=True)
                    sys.exit(2)
                if spectrum_error is not None and arm!='random':
                    record_spectrum_failure(rep,seed,arm,data['protocol_sha256'],spectrum_error)
                    total+=1
                    continue
                try:
                    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_ising_fresh_synthetic.py'),str(rep),str(seed),arm],cwd=ROOT,env=env,text=True,capture_output=True,timeout=remaining)
                except subprocess.TimeoutExpired as exc:
                    print(f'BATCH TIME LIMIT: corpus={rep} seed={seed} arm={arm}; run may need partial-artifact review: {exc}',flush=True)
                    sys.exit(2)
                if result.stdout: print(result.stdout.strip(),flush=True)
                if result.returncode:
                    print(f'RUN FAILED: corpus={rep} seed={seed} arm={arm} code={result.returncode}\n{result.stderr}',flush=True)
                    sys.exit(result.returncode)
                total+=1
    print(f'BATCH COMPLETE: runs={total}, wall_seconds={time.monotonic()-start:.2f}',flush=True)
