"""Run frozen shuffled controls sequentially with a shared 30-minute deadline."""
from pathlib import Path
import os
import json
import tempfile
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.train_text8_lexical_development import get_data
from word_energy.text8_factorial_models import ARMS,SEEDS
from word_energy.tracking import tracking_uri

def record_failure(experiment_id, arm, seed, reason, stdout, stderr):
    client=MlflowClient()
    matches=[r for r in client.search_runs([experiment_id],max_results=1000)
             if r.data.tags.get('arm')==arm and r.data.tags.get('seed')==str(seed)]
    if len(matches)!=1:
        print(f'Could not uniquely identify interrupted run: {arm}/{seed}, matches={len(matches)}',flush=True)
        return
    run=matches[0]
    def decode(value):
        if isinstance(value,bytes): return value.decode('utf-8',errors='replace')
        return value or ''
    payload={'reason':reason,'arm':arm,'seed':seed,'stdout':decode(stdout),'stderr':decode(stderr)}
    temp_root=ROOT/'.tmp'; temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as temporary:
        path=Path(temporary)/'failure.json'; path.write_text(json.dumps(payload,indent=2),encoding='utf-8')
        client.log_artifact(run.info.run_id,str(path),'interruption')
    client.set_tag(run.info.run_id,'interruption_reason',reason)
    if run.info.status=='RUNNING': client.set_terminated(run.info.run_id,status='KILLED')


if __name__=='__main__':
    mlflow.set_tracking_uri(tracking_uri()); get_data('6cf76591c3fa49c686a3d81d7f225607')
    exp=mlflow.set_experiment('word-energy-text8-frequency-shuffle-control')
    if MlflowClient().search_runs([exp.experiment_id],max_results=1000): raise RuntimeError('existing runs: inspect before restarting')
    env=os.environ.copy(); env.update({'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MLFLOW_DISABLE_AGENT_HINT':'1'})
    env.pop('MLFLOW_RUN_ID',None)
    start=time.monotonic(); deadline=start+1800; complete=0
    for seed in SEEDS:
        for arm in ('tied-frequency','role-frequency'):
            remaining=deadline-time.monotonic()
            if remaining<=0: raise TimeoutError(f'incomplete factorial: {complete}/6')
            try:
                result=subprocess.run([sys.executable,str(ROOT/'scripts/train_text8_frequency_shuffle.py'),arm,str(seed)],
                                      cwd=ROOT,env=env,text=True,capture_output=True,timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                record_failure(exp.experiment_id,arm,seed,'shared deadline exceeded',exc.stdout,exc.stderr)
                raise TimeoutError(f'incomplete factorial: {complete}/6; inspect interrupted {arm}/{seed}') from exc
            if result.stdout: print(result.stdout.strip(),flush=True)
            if result.returncode:
                record_failure(exp.experiment_id,arm,seed,f'child exit {result.returncode}',result.stdout,result.stderr)
                raise RuntimeError(f'failed {arm}/{seed}; inspect attempt: {result.stderr}')
            complete+=1
    print(f'SHUFFLE CONTROL COMPLETE runs={complete} wall_seconds={time.monotonic()-start:.2f}',flush=True)
