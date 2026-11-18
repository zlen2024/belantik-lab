"""Run exactly 24 frozen development training jobs sequentially."""
from pathlib import Path
import json,os,subprocess,sys,tempfile,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.build_text8_fresh_multi_data import load
from word_energy.tracking import tracking_uri

def failure(client,exp,arm,seed,shuffle,reason,stdout,stderr):
    matches=[r for r in client.search_runs([exp],max_results=1000) if r.data.tags.get('arm')==arm and r.data.tags.get('seed')==str(seed) and r.data.tags.get('shuffle')==str(shuffle)]
    if len(matches)!=1: print('Interrupted before unique MLflow run; reason:',reason,flush=True);return
    run=matches[0]; temp=ROOT/'.tmp';temp.mkdir(exist_ok=True)
    def decode(s):return s.decode('utf-8',errors='replace') if isinstance(s,bytes) else (s or '')
    with tempfile.TemporaryDirectory(dir=temp) as d:
        path=Path(d)/'failure.json';path.write_text(json.dumps({'reason':reason,'stdout':decode(stdout),'stderr':decode(stderr)},indent=2),encoding='utf-8');client.log_artifact(run.info.run_id,str(path),'interruption')
    client.set_tag(run.info.run_id,'interruption_reason',reason)
    if run.info.status=='RUNNING':client.set_terminated(run.info.run_id,status='KILLED')

if __name__=='__main__':
    data_id=sys.argv[1];mlflow.set_tracking_uri(tracking_uri());load(data_id);client=MlflowClient();exp=client.get_experiment_by_name('word-energy-text8-fresh-multi-shuffle')
    if any(r.data.tags.get('arm') for r in client.search_runs([exp.experiment_id],max_results=1000)):raise ValueError('existing training runs; inspect before restart')
    env=os.environ.copy();env.update({'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MLFLOW_DISABLE_AGENT_HINT':'1'});env.pop('MLFLOW_RUN_ID',None)
    start=time.monotonic();deadline=start+2700;n=0
    for seed in (17,29,43):
        for arm in ('tied-frequency','role-frequency','vector-frequency'):
            for shuffle in ((11331,11332) if arm!='vector-frequency' else (0,11330,11331,11332)):
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError(f'completed {n}/24; deadline before next job')
                try:
                    r=subprocess.run([sys.executable,str(ROOT/'scripts/train_text8_fresh_multi.py'),data_id,arm,str(seed),str(shuffle)],cwd=ROOT,env=env,text=True,capture_output=True,timeout=remaining)
                except subprocess.TimeoutExpired as e:
                    failure(client,exp.experiment_id,arm,seed,shuffle,'deadline exceeded',e.stdout,e.stderr);raise
                if r.stdout:print(r.stdout.strip(),flush=True)
                if r.returncode:
                    failure(client,exp.experiment_id,arm,seed,shuffle,f'exit {r.returncode}',r.stdout,r.stderr);raise RuntimeError(f'failure {arm}/{seed}/{shuffle}: {r.stderr}')
                n+=1;print(f'COMPLETED {n}/24',flush=True)
    print(f'COMPLETE {n}/24 wall_seconds={time.monotonic()-start:.2f}',flush=True)
