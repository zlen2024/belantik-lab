"""Run fixed and relative-W controls with a shared CPU time budget."""
from pathlib import Path
import os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
def main():
    env=os.environ.copy();env.update({'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','JAX_PLATFORMS':'cpu','MLFLOW_DISABLE_AGENT_HINT':'1'})
    env.pop('MLFLOW_RUN_ID',None);deadline=time.monotonic()+900
    for seed in (17,29,43):
        for arm in ('ising4','ising6'):
            for mode in ('fixed','redistribute'):
                remain=deadline-time.monotonic()
                if remain<=0:raise TimeoutError('15 minute budget exceeded')
                r=subprocess.run([sys.executable,str(ROOT/'scripts/train_spin_capacity_lexical.py'),arm,str(seed),mode],cwd=ROOT,env=env,timeout=remain)
                if r.returncode:raise RuntimeError(f'{arm}/{mode}/{seed} exited {r.returncode}')
if __name__=='__main__':main()
