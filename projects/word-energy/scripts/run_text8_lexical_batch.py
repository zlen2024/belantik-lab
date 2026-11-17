"""Execute the frozen twelve-run text8 panel under one 45-minute deadline."""
from pathlib import Path
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.train_text8_lexical_development import get_data
from word_energy.text8_lexical_models import ARMS,SEEDS
from word_energy.tracking import tracking_uri


if __name__=='__main__':
    data_run=sys.argv[1]; mlflow.set_tracking_uri(tracking_uri()); get_data(data_run)
    exp=MlflowClient().get_experiment_by_name('word-energy-text8-lexical-development')
    if exp is None: raise ValueError('dataset experiment missing')
    existing=[r for r in MlflowClient().search_runs([exp.experiment_id],max_results=1000) if 'arm' in r.data.tags]
    if existing: raise RuntimeError('existing model runs; inspect before any continuation')
    env=os.environ.copy(); env.update({'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MLFLOW_DISABLE_AGENT_HINT':'1'})
    env.pop('MLFLOW_RUN_ID',None)
    start=time.monotonic(); deadline=start+2700; completed=0
    for seed in SEEDS:
        for arm in ARMS:
            remaining=deadline-time.monotonic()
            if remaining<=0: raise TimeoutError(f'incomplete: {completed}/12 runs')
            try:
                result=subprocess.run([sys.executable,str(ROOT/'scripts/train_text8_lexical_development.py'),data_run,arm,str(seed)],
                                      cwd=ROOT,env=env,text=True,capture_output=True,timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f'incomplete: {completed}/12; inspect interrupted {arm}/{seed}') from exc
            if result.stdout: print(result.stdout.strip(),flush=True)
            if result.returncode:
                raise RuntimeError(f'failed {arm}/{seed}; preserve attempt: {result.stderr}')
            completed+=1
    print(f'BATCH COMPLETE: runs={completed}, wall_seconds={time.monotonic()-start:.2f}',flush=True)
