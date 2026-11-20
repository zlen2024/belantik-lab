"""Replace epoch two's repeated windows with a fresh sample at fixed update budget."""
from pathlib import Path
import argparse
import json
import sys
import tempfile

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_four_spin_full_vocab import ranks_for, DATA, SUMMARY, load, artifact_dir
from scripts.build_wikitext2_document_data import corpus_data, parse, sample
from scripts.build_wikitext2_joint_context_data import windows
from scripts.build_ewt_retrieval_windows import build_ewt
from scripts.train_text8_lexical_development import snapshot
from word_energy.wikitext_four_spin import initial, functions, SEEDS
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract, log_contract, tracking_uri

REFERENCES = {'wiki':'782220fbb92b4f18ac410c53e0e44ff4','ewt':'6219acde7e58415ba42ba9b067f7061e'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,choices=SEEDS)
    parser.add_argument("--arm",choices=("joint4","vector10"))
    args = parser.parse_args()
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    data, meta = load(DATA)
    vocab = meta['vocabulary']
    ids, docs = corpus_data(parse('train'),{v:i for i,v in enumerate(vocab)})
    pairs,positions,groups = sample(ids,docs,200000,np.random.default_rng(25101))
    new, pos, _, _, _ = windows(ids,docs,pairs,positions,groups,200000,25102)
    counts = data['counts']
    prior = .25*np.log(counts.astype(float))
    adjusted = prior + np.log(data['noise_q'])
    ewt, edoc, _ = build_ewt(ROOT,vocab)
    panels = {'wiki':(data['validation_windows'],data['validation_document']), 'ewt':(ewt,edoc)}
    def position_count(position):
        canonical = np.column_stack((position[:,0],np.sort(position[:,1:],axis=1)))
        return len(np.unique(canonical,axis=0))
    position_counts = [position_count(data['train_positions']),position_count(np.concatenate((data['train_positions'],pos)))]
    summary = json.loads((artifact_dir(client,SUMMARY)/'config.json').read_text('utf8'))
    mlflow.set_experiment('word-energy-fixed-compute-window-scale')
    for seed in ([args.seed] if args.seed else SEEDS):
        for arm in ([args.arm] if args.arm else ('joint4','vector10')):
            ref = summary['results'][f'{arm}:{seed}']['run_id']
            cfg = json.loads((artifact_dir(client,ref)/'config.json').read_text('utf8'))
            assert cfg['dataset_hashes']==meta['hashes']
            params = initial(arm,seed,len(vocab))
            _,_,_,step,opt = functions(arm,prior)
            state = opt.init(params)
            with mlflow.start_run(run_name=f'400k-window-{arm}-{seed}') as run:
                mlflow.log_params(dict(arm=arm,seed=seed,distinct_sample_slots=400000,training_presentations=600000,reference=ref))
                for path in (Path(__file__),ROOT/'notes/FOUR_SPIN_WINDOW_SCALE_PROTOCOL.md',ROOT/'scripts/evaluate_four_spin_full_vocab.py',ROOT/'scripts/build_ewt_retrieval_windows.py'):
                    mlflow.log_artifact(str(path),'source_snapshot')
                with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
                    for epoch,train in enumerate((data['train_windows'],new,data['train_windows'])):
                        for start in range(0,len(train),256):
                            ix = data['orders'][epoch,start:start+256]
                            params,state,loss = step(params,state,train[ix],data['train_noise'][epoch,ix])
                            if not np.isfinite(float(loss)):
                                raise ValueError('nonfinite loss')
                        if epoch==0 and {k:sha_array(np.asarray(v)) for k,v in params.items()} != cfg['checkpoint_hashes']['epoch1']:
                            with np.load(artifact_dir(client,ref)/'checkpoints/epoch1.npz') as old:
                                error = {k:float(np.max(np.abs(np.asarray(v)-old[k]))) for k,v in params.items()}
                            mlflow.log_metrics({'first_pass_error_'+k:v for k,v in error.items()})
                            raise ValueError(f'first pass mismatch {error}')
                    hashes = snapshot(params,'epoch3',Path(tmp))
                    metrics, arrays = {}, {}
                    params = {k:np.asarray(v) for k,v in params.items()}
                    for panel,(w,d) in panels.items():
                        rank,err = ranks_for(params,w,adjusted,arm)
                        with np.load(artifact_dir(client,REFERENCES[panel])/'ranks/ranks.npz') as old:
                            assert np.array_equal(d,old['doc'])
                            diff = 1/rank-1/old[f'{arm}_{seed}']
                        article = np.array([diff[d==j].mean() for j in np.unique(d)])
                        metrics[panel] = dict(mrr=float(np.mean(1/rank)),article_delta=float(article.mean()),
                            recall10=float((rank<=10).mean()),check_error=err)
                        mlflow.log_metric(panel+'.article_mrr_delta',float(article.mean()))
                        arrays[panel+'_ranks'] = rank
                        arrays[panel+'_article_delta'] = article
                    target=Path(tmp)/'scale_ranks.npz'
                    np.savez_compressed(target,**arrays)
                    mlflow.log_artifact(str(target),'evaluation')
                out=dict(metrics=metrics,arm=arm,seed=seed,source_checkpoint=ref,rank_references=REFERENCES,
                         data=DATA,new_window_hash=sha_array(new),original_hashes=meta['hashes'],checkpoint=hashes,
                         unique_train_positions=position_counts,first_epoch_hash_matches=True)
                log_contract(mlflow,RunContract('classical','cpu_gpu_simulator','window_scale_'+arm,'fixed scorer plus log noise prior',
                    '12 spins or shared vector','Ising beta1','exact score, Adam 3 passes batch 256',seed,sha_array(new),
                    'WikiText train, WikiText and EWT development','4 spins per word or 10D vector',
                    'same articles; reused dev panels; sampled windows overlap'),out)
                print(arm,seed,run.info.run_id,metrics,position_counts,flush=True)


if __name__=='__main__':
    main()
