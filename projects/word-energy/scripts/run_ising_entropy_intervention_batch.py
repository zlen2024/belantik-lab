"""Run the frozen targeted-diversity panel, stopping on any mismatch."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from word_energy.fresh_synthetic import make_fresh_corpus
from word_energy.ising.exact_training import prepare_data
from word_energy.ising.spectral_initializer import array_hash, spectral_biases
from word_energy.tracking import tracking_uri
from scripts.run_ising_fresh_synthetic import SEEDS

PERMUTATIONS=(7043,7019,7027)
ARMS=('baseline','early-bank','late-bank','early-song')


def old_runs(client):
    exp=client.get_experiment_by_name('word-energy-corpus227-permutation-panel-v1')
    if exp is None: raise ValueError('old experiment missing')
    result={}
    for run in client.search_runs([exp.experiment_id],max_results=1000):
        tags=run.data.tags
        if tags.get('permutation_seed') not in map(str,PERMUTATIONS) or tags.get('seed') not in map(str,SEEDS): continue
        key=(int(tags['permutation_seed']),int(tags['seed']))
        if key in result or run.info.status!='FINISHED': raise ValueError(f'invalid old run {key}')
        path=ROOT/'mlruns'/exp.experiment_id/run.info.run_id/'artifacts'/'config.json'
        result[key]=json.loads(path.read_text(encoding='utf-8'))
    if len(result)!=12: raise ValueError(f'expected 12 old runs, found {len(result)}')
    return result


def preflight(client):
    old=old_runs(client); data=prepare_data(corpus=make_fresh_corpus(227))
    spectral=spectral_biases(data['train_windows'],len(data['ids']))
    for (perm,seed),cfg in old.items():
        if cfg['protocol_sha256']!=data['protocol_sha256'] or cfg['schedule_sha256']!=data['schedule_sha256']:
            raise ValueError('old data or schedule mismatch')
        expected=spectral['biases'][np.random.default_rng(perm).permutation(len(data['ids']))]
        if array_hash(expected)!=cfg['initial_hashes']['biases']:
            raise ValueError('old initial bias mismatch')
    return old


def compare_baseline(client, exp_id, perm, seed, old):
    matches=[run for run in client.search_runs([exp_id],max_results=1000)
             if run.data.tags.get('permutation_seed')==str(perm) and run.data.tags.get('seed')==str(seed)
             and run.data.tags.get('arm')=='baseline']
    if len(matches)!=1 or matches[0].info.status!='FINISHED': raise ValueError('baseline missing or failed')
    path=ROOT/'mlruns'/exp_id/matches[0].info.run_id/'artifacts'/'config.json'
    new=json.loads(path.read_text(encoding='utf-8'))
    for name in ('stage1_initial_0','stage1_lexical_400','stage2_initial_0','stage2_joint_100'):
        for field in ('biases','internal','interaction'):
            if not np.array_equal(np.asarray(new['snapshots'][name][field]),np.asarray(old['snapshots'][name][field])):
                raise ValueError(f'baseline parity failure: {perm}:{seed}:{name}:{field}')
    for name in ('fixed_lexical400','fixed_joint100'):
        if new[name]!=old[name]: raise ValueError(f'baseline metric parity failure {name}')


if __name__=='__main__':
    mlflow.set_tracking_uri(tracking_uri()); client=MlflowClient(); old=preflight(client)
    exp=mlflow.set_experiment('word-energy-entropy-intervention-v1')
    if client.search_runs([exp.experiment_id],max_results=1000): raise RuntimeError('existing panel: inspect before rerunning')
    env=os.environ.copy(); env.update({'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MLFLOW_DISABLE_AGENT_HINT':'1'})
    start=time.monotonic(); deadline=start+2700; total=0
    for perm in PERMUTATIONS:
        for seed in SEEDS:
            for arm in ARMS:
                remaining=deadline-time.monotonic()
                if remaining<=0: raise TimeoutError(f'panel incomplete: {total}/48')
                try:
                    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_ising_entropy_intervention_arm.py'),str(seed),str(perm),arm],cwd=ROOT,env=env,text=True,capture_output=True,timeout=remaining)
                except subprocess.TimeoutExpired as exc: raise TimeoutError(f'panel incomplete: {total}/48; inspect interrupted run') from exc
                if result.stdout: print(result.stdout.strip(),flush=True)
                if result.returncode: raise RuntimeError(f'failed {perm}:{seed}:{arm}: {result.stderr}')
                total+=1
                if arm=='baseline': compare_baseline(client,exp.experiment_id,perm,seed,old[(perm,seed)])
    print(f'BATCH COMPLETE: runs={total}, wall_seconds={time.monotonic()-start:.2f}',flush=True)
