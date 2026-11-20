"""Fixed-budget matched branches sharing the identical first-pass model and Adam state."""
import sys
from pathlib import Path
import tempfile

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_four_spin_window_scale import (
    DATA, REFERENCES, SEEDS, artifact_dir, build_ewt, corpus_data, functions, initial,
    load, log_contract, parse, ranks_for, RunContract, sample, sha_array, snapshot,
    tracking_uri, windows,
)


def train_epoch(step, params, state, train, data, epoch):
    for start in range(0,len(train),256):
        ix = data['orders'][epoch,start:start+256]
        params,state,loss = step(params,state,train[ix],data['train_noise'][epoch,ix])
        if not np.isfinite(float(loss)):
            raise FloatingPointError('nonfinite loss')
    return params,state


def main():
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    data,meta = load(DATA)
    a = data['train_windows']
    vocab = meta['vocabulary']
    ids,docs = corpus_data(parse('train'),{w:i for i,w in enumerate(vocab)})
    pairs,positions,groups = sample(ids,docs,200000,np.random.default_rng(25101))
    b,pos,_,_,_ = windows(ids,docs,pairs,positions,groups,200000,25102)
    def unique_count(p):
        return len(np.unique(np.column_stack((p[:,0],np.sort(p[:,1:],axis=1))),axis=0))
    position_counts = [unique_count(data['train_positions']),unique_count(np.vstack((data['train_positions'],pos)))]
    ewt,edoc,_ = build_ewt(ROOT,vocab)
    panels = {'wiki':(data['validation_windows'],data['validation_document']),'ewt':(ewt,edoc)}
    prior = .25*np.log(data['counts'].astype(float))
    adjusted = prior + np.log(data['noise_q'])
    mlflow.set_experiment('word-energy-window-scale-shared-fork')
    for seed in SEEDS:
        for arm in ('joint4','vector10'):
            with mlflow.start_run(run_name=f'paired-{arm}-{seed}') as run:
                mlflow.log_params(dict(arm=arm,seed=seed,sample_slots='200000,400000',presentations=600000))
                for p in (Path(__file__), ROOT/'notes/FOUR_SPIN_WINDOW_SCALE_FORK.md',ROOT/'notes/FOUR_SPIN_WINDOW_SCALE_PROTOCOL.md',ROOT/'scripts/run_four_spin_window_scale.py',ROOT/'scripts/evaluate_four_spin_full_vocab.py',ROOT/'scripts/build_ewt_retrieval_windows.py'):
                    mlflow.log_artifact(str(p),'source_snapshot')
                _,_,_,step,opt = functions(arm,prior)
                init = initial(arm,seed,len(prior))
                shared,adam = train_epoch(step,init,opt.init(init),a,data,0)
                import jax
                state_hashes = [sha_array(np.asarray(v)) for v in jax.tree_util.tree_leaves((shared,adam))]
                arrays,metrics,final_hashes = {},{},{}
                for branch,middle in [('repeat',a),('fresh',b)]:
                    params,state = train_epoch(step,shared,adam,middle,data,1)
                    params,state = train_epoch(step,params,state,a,data,2)
                    with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
                        final_hashes[branch] = snapshot(params,branch,Path(tmp))
                    params = {k:np.asarray(v) for k,v in params.items()}
                    for panel,(w,d) in panels.items():
                        rank,err = ranks_for(params,w,adjusted,arm)
                        arrays[branch+panel] = rank
                        if branch=='repeat':
                            with np.load(artifact_dir(client,REFERENCES[panel])/'ranks/ranks.npz') as old:
                                assert np.array_equal(d,old['doc'])
                                metrics[panel] = {'historical_changed_ranks':int(np.sum(old[f'{arm}_{seed}']!=rank))}
                        else:
                            rr = 1/rank-1/arrays['repeat'+panel]
                            article = np.array([rr[d==j].mean() for j in np.unique(d)])
                            arrays[panel+'_article_delta'] = article
                            metrics[panel].update(article_delta=float(article.mean()),mrr=float(np.mean(1/rank)),recall10=float((rank<=10).mean()),source_error=err)
                            mlflow.log_metric(panel+'.article_mrr_delta',float(article.mean()))
                with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
                    p=Path(tmp)/'scale_ranks.npz'
                    np.savez_compressed(p,**arrays)
                    mlflow.log_artifact(str(p),'evaluation')
                assert state_hashes==[sha_array(np.asarray(v)) for v in jax.tree_util.tree_leaves((shared,adam))]
                log_contract(mlflow,RunContract('classical','cpu_gpu_simulator','paired_window_scale_'+arm,'fixed corrected full-vocabulary ranking','12 spins or vector10','Ising beta1','exact factorization + Adam',seed,sha_array(b),'WikiText train and WikiText/EWT dev','3 K4 or 10D vector','same corpus and repeated development panels'),
                    dict(metrics=metrics,arm=arm,seed=seed,shared_state_hashes=state_hashes,final_hashes=final_hashes,
                         data=DATA,original_hashes=meta['hashes'],rank_references=REFERENCES,unique_train_positions=position_counts,new_window_hash=sha_array(b)))
                print(run.info.run_id,arm,seed,metrics,flush=True)


if __name__=='__main__':
    main()
