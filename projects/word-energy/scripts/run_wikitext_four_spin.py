"""Run frozen four-spin panel with 45-minute shared deadline."""
from pathlib import Path
import json,os,subprocess,sys,tempfile,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.build_wikitext2_joint_context_data import load
from word_energy.wikitext_four_spin import ARMS,SEEDS
from word_energy.tracking import tracking_uri
DATA='f1b7db5231ed4a5788b566e388f8e731';EXP='word-energy-wikitext2-four-spin-context'

def failure(c,exp,arm,seed,reason,out,err):
 rows=[r for r in c.search_runs([exp],max_results=1000) if r.data.tags.get('arm')==arm and r.data.tags.get('seed')==str(seed)]
 if len(rows)!=1:print(f'No unique failed run {arm}/{seed}: {reason}',flush=True);return
 run=rows[0];tmp=ROOT/'.tmp';tmp.mkdir(exist_ok=True)
 def dec(x):return x.decode('utf8',errors='replace') if isinstance(x,bytes) else (x or '')
 with tempfile.TemporaryDirectory(dir=tmp) as d:
  p=Path(d)/'failure.json';p.write_text(json.dumps({'reason':reason,'stdout':dec(out),'stderr':dec(err)},indent=2),encoding='utf8');c.log_artifact(run.info.run_id,str(p),'interruption')
 c.set_tag(run.info.run_id,'interruption_reason',reason)
 if run.info.status=='RUNNING':c.set_terminated(run.info.run_id,status='KILLED')

if __name__=='__main__':
 mlflow.set_tracking_uri(tracking_uri());load(DATA);c=MlflowClient();exp=mlflow.set_experiment(EXP)
 if any(r.data.tags.get('arm') for r in c.search_runs([exp.experiment_id],max_results=1000)):raise ValueError('existing attempts; inspect first')
 env=os.environ.copy();env.update({'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MLFLOW_DISABLE_AGENT_HINT':'1','JAX_PLATFORMS':'cpu'});env.pop('MLFLOW_RUN_ID',None)
 start=time.monotonic();deadline=start+2700;complete=0
 for seed in SEEDS:
  for arm in ARMS:
   rem=deadline-time.monotonic()
   if rem<=0:raise TimeoutError(f'completed {complete}/12; deadline')
   try:r=subprocess.run([sys.executable,str(ROOT/'scripts/train_wikitext_four_spin.py'),arm,str(seed)],cwd=ROOT,env=env,text=True,capture_output=True,timeout=rem)
   except subprocess.TimeoutExpired as e:failure(c,exp.experiment_id,arm,seed,'deadline exceeded',e.stdout,e.stderr);raise
   if r.stdout:print(r.stdout.strip(),flush=True)
   if r.returncode:failure(c,exp.experiment_id,arm,seed,f'exit {r.returncode}',r.stdout,r.stderr);raise RuntimeError(f'{arm}/{seed}: {r.stderr}')
   complete+=1;print(f'COMPLETE {complete}/12',flush=True)
 print(f'COMPLETE 12/12 wall_seconds={time.monotonic()-start:.2f}',flush=True)
