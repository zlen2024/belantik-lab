"""Continue an interrupted frozen panel without discarding its first attempt."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from word_energy.tracking import tracking_uri
from scripts.run_ising_entropy_intervention_batch import PERMUTATIONS,ARMS,preflight,compare_baseline
from scripts.run_ising_fresh_synthetic import SEEDS

INTERRUPTED='b9af12890f824290b86ec81b7ac09d56'
INTERRUPTED_KEY=(7043,269,'baseline')
SOURCE_ROOT=ROOT/'mlruns'/'10'/'89be4d294b7d4a1ca9e0251461fd6c9c'/'artifacts'/'source_snapshot'
SOURCE_FILES=('word_energy/fresh_synthetic.py','word_energy/ising/spectral_initializer.py',
              'word_energy/ising/exact_training.py','scripts/diagnose_ising_validation.py',
              'word_energy/ising/entropy_intervention.py','scripts/diagnose_corpus227_permutation_dynamics.py')


def runs_by_key(client,exp_id):
    result={}
    for run in client.search_runs([exp_id],max_results=1000):
        t=run.data.tags
        if not all(k in t for k in ('permutation_seed','seed','arm')): continue
        key=(int(t['permutation_seed']),int(t['seed']),t['arm'])
        result.setdefault(key,[]).append(run)
    return result


def validate(client,exp_id,old):
    for rel in SOURCE_FILES:
        path=ROOT/rel
        if hashlib.sha256(path.read_bytes()).digest()!=hashlib.sha256((SOURCE_ROOT/path.name).read_bytes()).digest():
            raise ValueError(f'source differs from initial runs: {rel}')
    grouped=runs_by_key(client,exp_id)
    expected={(p,s,a) for p in PERMUTATIONS for s in SEEDS for a in ARMS}
    if set(grouped)-expected: raise ValueError('unexpected run identities')
    for key,runs in grouped.items():
        finished=[r for r in runs if r.info.status=='FINISHED']
        other=[r for r in runs if r.info.status!='FINISHED']
        if len(finished)>1 or len(other)>1: raise ValueError(f'duplicate attempts: {key}')
        if other and (key!=INTERRUPTED_KEY or other[0].info.run_id!=INTERRUPTED or other[0].info.status!='KILLED'):
            raise ValueError(f'unexpected unfinished attempt: {key}')
        if finished and other and finished[0].data.tags.get('retry_of_run_id')!=other[0].info.run_id:
            raise ValueError(f'unlinked retry: {key}')
        if finished:
            run=finished[0]; path=ROOT/'mlruns'/exp_id/run.info.run_id/'artifacts'/'config.json'
            cfg=json.loads(path.read_text(encoding='utf-8'))
            prior=old[(key[0],key[1])]
            for field in ('protocol_sha256','schedule_sha256','train_token_hash','initial_hashes','lexical_config','joint_config'):
                if cfg[field]!=prior[field]: raise ValueError(f'provenance mismatch {key}: {field}')
            if cfg['snapshots']['stage1_lexical_400']!=cfg['snapshots']['stage2_initial_0']:
                raise ValueError(f'handoff mismatch {key}')
            if key[2]=='baseline': compare_baseline(client,exp_id,key[0],key[1],old[(key[0],key[1])])
    return grouped


if __name__=='__main__':
    mlflow.set_tracking_uri(tracking_uri()); client=MlflowClient(); old=preflight(client)
    exp=client.get_experiment_by_name('word-energy-entropy-intervention-v1')
    if exp is None: raise ValueError('experiment missing')
    grouped=validate(client,exp.experiment_id,old)
    env=os.environ.copy(); env.update({'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MLFLOW_DISABLE_AGENT_HINT':'1'})
    env.pop('WORD_ENERGY_RETRY_OF',None)
    env.pop('MLFLOW_RUN_ID',None)
    start=time.monotonic(); deadline=start+2700; newly_finished=0
    for perm in PERMUTATIONS:
        for seed in SEEDS:
            for arm in ARMS:
                key=(perm,seed,arm); existing=grouped.get(key,[])
                if any(r.info.status=='FINISHED' for r in existing): continue
                remaining=deadline-time.monotonic()
                if remaining<=0: raise TimeoutError(f'continuation deadline, completed {newly_finished} new runs')
                child_env=env.copy()
                if key==INTERRUPTED_KEY: child_env['WORD_ENERGY_RETRY_OF']=INTERRUPTED
                try:
                    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_ising_entropy_intervention_arm.py'),str(seed),str(perm),arm],cwd=ROOT,env=child_env,text=True,capture_output=True,timeout=remaining)
                except subprocess.TimeoutExpired as exc:
                    raise TimeoutError(f'continuation deadline; inspect interrupted attempt {key}') from exc
                if result.stdout: print(result.stdout.strip(),flush=True)
                if result.returncode: raise RuntimeError(f'failed {key}: {result.stderr}')
                newly_finished+=1
                if arm=='baseline': compare_baseline(client,exp.experiment_id,perm,seed,old[(perm,seed)])
    print(f'CONTINUATION COMPLETE new_runs={newly_finished} wall_seconds={time.monotonic()-start:.2f}',flush=True)
