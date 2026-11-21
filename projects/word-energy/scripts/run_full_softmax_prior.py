"""Train the frozen full-softmax offset ablation and compare paired ranks."""
from pathlib import Path
import json
import os
import sys
import tempfile
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.build_wikitext2_joint_context_data import load
from scripts.build_ewt_retrieval_windows import build_ewt
from scripts.evaluate_four_spin_full_vocab import ranks_for
from scripts.run_full_softmax_pilot import DATA, make_functions
from word_energy.ising.exact_training import _imports
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract, log_contract, tracking_uri
from word_energy.wikitext_four_spin import initial, score_windows

PREVIOUS = {17: '30c212437ae840bebf7e22fc8294c420',
            29: '1531b87bf41048ea89dc796970d007f4',
            43: '77636915e34a45768e7ea01da7d940a9'}


def main(seed):
    jax, _, _ = _imports()
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    data, meta = load(DATA)
    offset = np.log(data['counts'].astype(float))
    params = initial('joint4', seed, len(offset))
    logits, loss_grad, step, optimizer = make_functions(offset)
    probe = data['train_windows'][:8]
    scores = np.asarray(logits(params, probe))
    for candidates in (probe[:, 0], (probe[:, 0] + 713) % len(offset)):
        expected = np.asarray(score_windows(params, np.column_stack((candidates, probe[:, 1:])), offset, 'joint4'))
        np.testing.assert_allclose(scores[np.arange(len(probe)), candidates], expected, atol=1e-11, rtol=0)
    value, gradient = loss_grad(params, probe)
    if not np.isfinite(float(value)) or any(not np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(gradient)):
        raise FloatingPointError('nonfinite initial loss or gradient')
    ewt, ewt_docs, ewt_meta = build_ewt(ROOT, meta['vocabulary'])
    panels = {'wiki': (data['validation_windows'], data['validation_document']), 'ewt': (ewt, ewt_docs)}
    mlflow.set_experiment('word-energy-full-softmax-prior-ablation')
    with mlflow.start_run(run_name=f'full-softmax-log-count-seed{seed}') as run:
        mlflow.set_tags({'seed': str(seed), 'dataset_run_id': DATA, 'test_status': 'reserved closed'})
        mlflow.log_params({'seed': seed, 'batch': 256, 'epochs': 3, 'lr': .01, 'offset': 'log(count)',
                           'objective': 'full-vocabulary cross-entropy', 'vocabulary': len(offset)})
        for path in (Path(__file__), ROOT/'scripts/run_full_softmax_pilot.py',
                     ROOT/'notes/FULL_SOFTMAX_PRIOR_ABLATION_PROTOCOL.md', ROOT/'word_energy/wikitext_four_spin.py'):
            mlflow.log_artifact(str(path), 'source_snapshot')
        state = optimizer.init(params)
        train = data['train_windows']
        start = time.monotonic()
        for epoch in range(3):
            total = 0.
            for i in range(0, len(train), 256):
                indices = data['orders'][epoch, i:i+256]
                params, state, value = step(params, state, train[indices])
                value = float(value)
                if not np.isfinite(value):
                    raise FloatingPointError('nonfinite training loss')
                total += len(indices) * value
            mlflow.log_metric('train_cross_entropy', total / len(train), step=epoch+1)
            print(seed, 'epoch', epoch+1, 'loss', total/len(train), flush=True)
        seconds = time.monotonic() - start
        params = {k: np.asarray(v) for k, v in params.items()}
        arrays, results = {}, {}
        with np.load(artifact_dir(client, PREVIOUS[seed])/'results/ranks.npz') as previous:
            for panel, (windows, docs) in panels.items():
                if not np.array_equal(docs, previous[panel+'_docs']):
                    raise ValueError('document alignment mismatch')
                rank, error = ranks_for(params, windows, offset, 'joint4')
                baseline = previous[panel+'_raw'].copy()
                corrected = previous[panel+'_corrected'].copy()
                margin = previous[panel+'_baseline'].copy()
                prior_rank, _ = ranks_for(None, windows, offset, 'prior')
                arrays.update({panel+'_new': rank, panel+'_old': baseline, panel+'_old_corrected': corrected,
                               panel+'_margin': margin, panel+'_prior': prior_rank, panel+'_docs': docs})
                article_delta = np.array([(1/rank[docs == d] - 1/baseline[docs == d]).mean() for d in np.unique(docs)])
                results[panel] = {'article_mrr_delta': float(article_delta.mean()), 'mrr': float(np.mean(1/rank)),
                                  'recall10': float(np.mean(rank <= 10)), 'score_parity_max_error': error}
                mlflow.log_metric(panel+'.article_mrr_delta', float(article_delta.mean()))
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
            np.savez_compressed(Path(tmp)/'checkpoint.npz', **params)
            np.savez_compressed(Path(tmp)/'ranks.npz', **arrays)
            mlflow.log_artifacts(tmp, 'results')
        log_contract(mlflow, RunContract('classical', 'cpu_gpu_simulator', 'joint4_full_softmax_log_count',
            'normalized Ising joint score plus log count', '12 binary spins', 'beta=1',
            'exact factorization, full vocabulary, Adam', seed, sha_array(train),
            'WikiText train and WikiText/EWT development', '3 K4 and 8 spokes',
            'reused development panels; exploratory; no physical hardware'),
            {'dataset_run_id': DATA, 'dataset_hashes': meta['hashes'], 'previous_run': PREVIOUS[seed],
             'ewt': ewt_meta, 'results': results, 'training_seconds': seconds,
             'checkpoint_hashes': {k: sha_array(v) for k, v in params.items()}})
        mlflow.log_metric('training_seconds', seconds)
        print('FINISHED', run.info.run_id, seed, json.dumps(results), flush=True)


if __name__ == '__main__':
    selected_seed = int(sys.argv[1])
    if selected_seed not in PREVIOUS:
        raise ValueError('seed outside protocol')
    main(selected_seed)
