"""Collect all six verified runs; resample documents after averaging paired seeds."""
from pathlib import Path
import json
import sys

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_four_spin_paired_readout import artifact_dir, tracking_uri, SEEDS
from word_energy.tracking import RunContract, log_contract


def main():
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    experiment = mlflow.set_experiment('word-energy-window-scale-shared-fork')
    all_runs = client.search_runs([experiment.experiment_id],max_results=100)
    runs = [r for r in all_runs if r.info.status=='FINISHED' and 'arm' in r.data.params]
    expected = {(arm,s) for arm in ('joint4','vector10') for s in SEEDS}
    indexed = {(r.data.params['arm'],int(r.data.params['seed'])):r for r in runs}
    if set(indexed)!=expected or len(runs)!=6:
        raise ValueError(f'expected six verified runs; found {list(indexed)}')
    old = client.get_experiment_by_name('word-energy-fixed-compute-window-scale')
    attempts = client.search_runs([old.experiment_id])
    failures = [r.info.run_id for r in attempts if r.info.status=='FAILED']
    effects, config = {}, {}
    for arm in ('joint4','vector10'):
        for panel in ('wiki','ewt'):
            doc_diffs = []
            for seed in SEEDS:
                rid = indexed[arm,seed].info.run_id
                root = artifact_dir(client,rid)
                cfg = json.loads((root/'config.json').read_text('utf8'))
                assert cfg['shared_state_hashes'] and set(cfg['final_hashes']) == {'fresh','repeat'}
                config[f'{arm}_{seed}'] = dict(id=rid,metrics=cfg['metrics'],unique=cfg['unique_train_positions'])
                with np.load(root/'evaluation/scale_ranks.npz') as z:
                    doc_diffs.append(z[panel+'_article_delta'])
            d = np.mean(doc_diffs,axis=0)
            ix = np.random.default_rng(23140).integers(len(d),size=(1000,len(d)))
            ci = np.percentile(d[ix].mean(axis=1),[2.5,97.5]).tolist()
            effects[f'{arm}_{panel}'] = dict(delta=float(d.mean()),ci=ci,seed_delta=np.mean(doc_diffs,axis=1).tolist())
    primary=effects['joint4_wiki']
    passed=bool(primary['ci'][0]>0 and min(primary['seed_delta'])>0)
    out=dict(effects=effects,runs=config,earlier_failed_runs=failures,passed=passed)
    with mlflow.start_run(run_name='window-scale-decision') as run:
        log_contract(mlflow,RunContract('classical','cpu_gpu_simulator','window_scale_decision','paired article MRR','4 spins or vector','Ising beta1','article bootstrap',23140,cfg['new_window_hash'],'WikiText/EWT development','fixed size','conditional on three trained seeds, same articles'),out)
        mlflow.log_metric('wiki_joint_article_mrr_delta',primary['delta'])
        mlflow.log_artifact(str(Path(__file__)),'source_snapshot')
        print(run.info.run_id,json.dumps(out),flush=True)


if __name__=='__main__':
    main()
