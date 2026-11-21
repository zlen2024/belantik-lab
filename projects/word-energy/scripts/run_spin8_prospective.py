"""Run six fits sequentially; open holdout only after all six complete."""
from pathlib import Path
import os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
def main(data_id):
    env=os.environ.copy();env.update({'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','JAX_PLATFORMS':'cpu','MLFLOW_DISABLE_AGENT_HINT':'1'})
    env.pop('MLFLOW_RUN_ID',None);deadline=time.monotonic()+900
    for seed in (17,29,43):
        for arm in ('ising6','ising8'):
            remain=deadline-time.monotonic()
            if remain<=0:raise TimeoutError('overall 15 minute budget')
            p=subprocess.run([sys.executable,str(ROOT/'scripts/train_spin8_prospective.py'),data_id,arm,str(seed)],cwd=ROOT,env=env,timeout=remain)
            if p.returncode:raise RuntimeError(f'{arm}/{seed} failed: {p.returncode}')
    remain=deadline-time.monotonic()
    if remain<=0:raise TimeoutError('no time left for evaluation')
    p=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_spin8_prospective.py'),data_id],cwd=ROOT,env=env,timeout=remain)
    if p.returncode:raise RuntimeError(f'evaluation failed: {p.returncode}')
if __name__=='__main__':main(sys.argv[1])
