"""Score the entire fixed vocabulary in batches and verify against production scorers."""
from pathlib import Path
import sys
import json
import argparse

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_four_spin_paired_readout import DATA, SUMMARY, artifact_dir, load
from word_energy.wikitext_four_spin import state_table, unary_probabilities, score_windows, SEEDS
from word_energy.ising.exact_training import _imports
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract, log_contract, tracking_uri


def ranks_for(params, windows, prior, arm):
    jax, jnp, _ = _imports()
    if arm in ('joint4', 'pair_add4'):
        states = np.asarray(state_table(jnp))
        g = np.asarray(1.5 * jax.nn.sigmoid(params['interaction']) / 2)
        q = np.asarray(unary_probabilities(params))
        messages = q @ np.expm1((states*g) @ states.T).T
    ranks = []
    max_error = 0.
    for start in range(0,len(windows),128):
        center,left,right = windows[start:start+128].T
        rows = np.arange(len(center))
        if arm == 'prior':
            scores = np.broadcast_to(prior,(len(center),len(prior)))
        elif arm == 'vector10':
            embeddings = params['shared']
            scores = ((embeddings[left]+embeddings[right])*.5) @ embeddings.T + prior
        else:
            a,b = messages[left],messages[right]
            if arm == 'joint4':
                scores = np.log1p((a+b+a*b) @ q.T) + prior
            else:
                scores = np.log1p(a @ q.T) + np.log1p(b @ q.T) + prior
        if not np.isfinite(scores).all():
            raise ValueError('nonfinite full-vocabulary scores')
        if params is not None and start % 1024 == 0:
            for candidate in (center,(center+713)%len(prior)):
                observed = np.asarray(score_windows(params,np.column_stack((candidate,left,right)),prior,arm))
                max_error = max(max_error,float(np.abs(observed-scores[rows,candidate]).max()))
        correct = scores[rows,center,None]
        rank = 1+(scores>correct).sum(axis=1)+.5*((scores==correct).sum(axis=1)-1)
        ranks.extend(rank)
    if max_error > 1e-11:
        raise ValueError(f'source score parity failed: {max_error}')
    return np.asarray(ranks),max_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore-noise-prior', action='store_true')
    parser.add_argument('--ewt', action='store_true')
    args = parser.parse_args()
    restore = args.restore_noise_prior
    if args.ewt and not restore:
        parser.error('EWT protocol requires the fixed noise correction')
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment('word-energy-full-vocabulary-retrieval')
    client = MlflowClient()
    arrays,meta = load(DATA)
    summary = json.loads((artifact_dir(client,SUMMARY)/'config.json').read_text('utf8'))
    assert meta['hashes'] == summary['dataset_hashes']
    prior = .25*np.log(arrays['counts'].astype(float))
    correction = np.log(arrays['noise_q']) if restore else np.zeros_like(prior)
    windows,doc = arrays['validation_windows'],arrays['validation_document']
    panel = {'windows_sha256':meta['hashes']['validation_windows']}
    if args.ewt:
        from scripts.build_ewt_retrieval_windows import build_ewt
        windows,doc,panel = build_ewt(ROOT,meta['vocabulary'])
    docs = np.unique(doc)
    results, per_doc, all_ranks = {}, {}, {}
    with mlflow.start_run(run_name=('ewt-fixed-replication' if args.ewt else 'fixed-log-noise-correction' if restore else 'frozen-2000-candidate-retrieval')) as run:
        for p in (Path(__file__), ROOT/'notes/FOUR_SPIN_FULL_VOCAB_PROTOCOL.md'):
            mlflow.log_artifact(str(p),'source_snapshot')
        if args.ewt:
            for p in (ROOT/'notes/FOUR_SPIN_EWT_PROTOCOL.md',ROOT/'scripts/build_ewt_retrieval_windows.py'):
                mlflow.log_artifact(str(p),'source_snapshot')
        for arm in ('prior','joint4','pair_add4','vector10'):
            for seed in ((0,) if arm=='prior' else SEEDS):
                params,source = None,None
                if arm != 'prior':
                    trained = 'joint4' if arm=='pair_add4' else arm
                    source = summary['results'][f'{trained}:{seed}']['run_id']
                    directory = artifact_dir(client,source)
                    cfg = json.loads((directory/'config.json').read_text('utf8'))
                    with np.load(directory/'checkpoints/epoch3.npz',allow_pickle=False) as z:
                        params = dict(z)
                    assert cfg['dataset_run_id']==DATA and cfg['offset_sha256']==sha_array(prior)
                    assert cfg['checkpoint_hashes']['epoch3']=={k:sha_array(v) for k,v in params.items()}
                rank,error = ranks_for(params,windows,prior + correction,arm)
                key = f'{arm}_{seed}'
                all_ranks[key] = rank
                rr = 1/rank
                per_doc[key] = np.array([rr[doc==d].mean() for d in docs])
                results[key] = dict(mrr=float(rr.mean()),article_mrr=float(per_doc[key].mean()),
                                    recall10=float((rank<=10).mean()),check_max_error=error, source=source)
                mlflow.log_metric(key+'.mrr',results[key]['mrr'])
        rng = np.random.default_rng(23140)
        draws = rng.integers(len(docs),size=(1000,len(docs)))
        effects = {}
        for baseline in ('pair_add4','vector10','prior'):
            diff = np.mean([per_doc[f'joint4_{s}']-per_doc[f'{baseline}_{0 if baseline=="prior" else s}'] for s in SEEDS],axis=0)
            effects[baseline] = [float(diff.mean()),*np.percentile(diff[draws].mean(axis=1),[2.5,97.5]).tolist()]
        out = dict(panel=panel,restore_noise_prior=restore,results=results,effects=effects,hashes=meta['hashes'],data=DATA,summary=SUMMARY,
                   documents=len(docs),windows=len(windows),candidates=len(prior))
        log_contract(mlflow,RunContract('classical','cpu_gpu_simulator','full_vocabulary_ranking',
            'higher joint or pair score plus center prior','12 binary spins or shared 10D vector','beta=1',
            'exact factorization, all vocabulary candidates',23140,panel['windows_sha256'],
            ('UD EWT development' if args.ewt else 'WikiText-2 reused validation'),'3 K4 with 8 spokes','exploratory; exact observed word retrieval'),out)
        import tempfile
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
            p = Path(tmp)/'ranks.npz'
            np.savez_compressed(p,**all_ranks,doc=doc)
            mlflow.log_artifact(str(p),'ranks')
        print(run.info.run_id,json.dumps(out['results']),json.dumps(effects),flush=True)


if __name__ == '__main__':
    main()
