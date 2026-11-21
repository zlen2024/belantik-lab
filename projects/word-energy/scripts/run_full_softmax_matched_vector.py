"""Full-softmax tied-vector control matched to the four-spin ablation."""
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
from scripts.run_full_softmax_prior import DATA
from word_energy.ising.exact_training import _imports
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract, log_contract, tracking_uri
from word_energy.wikitext_four_spin import initial, score_windows
from word_energy.wikitext_joint_evaluation import canonical_scores

ISING = {17: '34ad840584164e5dac2059ca6ef5267b',
         29: '6fd4829031424eba917b99c5fd1ab9ca',
         43: '2ee7dde861fa4e82b960c0767e4bef6b'}


def make_functions(offset):
    jax, jnp, optax = _imports()
    offset = jnp.asarray(offset)

    def logits(params, windows):
        embedding = params['shared']
        contexts = (embedding[windows[:, 1]] + embedding[windows[:, 2]]) * .5
        return contexts @ embedding.T + offset

    def loss(params, windows):
        scores = logits(params, windows)
        return -jnp.mean(jax.nn.log_softmax(scores)[jnp.arange(len(windows)), windows[:, 0]])

    optimizer = optax.chain(optax.clip_by_global_norm(5.), optax.adam(.01))

    @jax.jit
    def step(params, state, windows):
        value, gradient = jax.value_and_grad(loss)(params, windows)
        updates, state = optimizer.update(gradient, state, params)
        return optax.apply_updates(params, updates), state, value

    return jax.jit(logits), jax.jit(jax.value_and_grad(loss)), step, optimizer


def main(seed):
    jax, _, _ = _imports()
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    data, meta = load(DATA)
    offset = np.log(data['counts'].astype(float))
    params = initial('vector10', seed, len(offset))
    logits, value_grad, step, optimizer = make_functions(offset)
    probe = data['train_windows'][:8]
    scores = np.asarray(logits(params, probe))
    for candidates in (probe[:, 0], (probe[:, 0]+713) % len(offset)):
        expected = np.asarray(score_windows(params, np.column_stack((candidates, probe[:, 1:])), offset, 'vector10'))
        np.testing.assert_allclose(scores[np.arange(len(probe)), candidates], expected, atol=1e-11, rtol=0)
    value, gradient = value_grad(params, probe)
    if not np.isfinite(float(value)) or any(not np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(gradient)):
        raise FloatingPointError('nonfinite initial loss or gradient')
    previous_root = artifact_dir(client, ISING[seed])
    config = json.loads((previous_root/'config.json').read_text('utf8'))
    if config['dataset_hashes'] != meta['hashes']:
        raise ValueError('Ising dataset hash mismatch')
    with np.load(previous_root/'results/checkpoint.npz', allow_pickle=False) as checkpoint:
        ising_params = dict(checkpoint)
    if {k: sha_array(v) for k, v in ising_params.items()} != config['checkpoint_hashes']:
        raise ValueError('Ising checkpoint hash mismatch')
    ewt, ewt_docs, ewt_meta = build_ewt(ROOT, meta['vocabulary'])
    if config['ewt']['windows_sha256'] != ewt_meta['windows_sha256'] or config['ewt']['groups_sha256'] != ewt_meta['groups_sha256']:
        raise ValueError('EWT panel mismatch')
    panels = {'wiki': (data['validation_windows'], data['validation_document']), 'ewt': (ewt, ewt_docs)}
    mlflow.set_experiment('word-energy-full-softmax-matched-vector')
    with mlflow.start_run(run_name=f'full-softmax-vector10-seed{seed}') as run:
        mlflow.set_tags({'seed': str(seed), 'test_status': 'reserved closed', 'ising_run': ISING[seed]})
        mlflow.log_params({'seed': seed, 'batch': 256, 'epochs': 3, 'lr': .01, 'offset': 'log(count)',
                           'objective': 'full-vocabulary cross-entropy', 'vocabulary': len(offset), 'parameters': 20000})
        for path in (Path(__file__), ROOT/'notes/FULL_SOFTMAX_MATCHED_VECTOR_PROTOCOL.md', ROOT/'word_energy/wikitext_four_spin.py'):
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
        seconds = time.monotonic()-start
        params = {k: np.asarray(v) for k, v in params.items()}
        arrays, results = {}, {}
        with np.load(previous_root/'results/ranks.npz') as old:
            for panel, (windows, docs) in panels.items():
                if not np.array_equal(docs, old[panel+'_docs']):
                    raise ValueError('document alignment mismatch')
                rank, error = ranks_for(params, windows, offset, 'vector10')
                # Recompute Ising ranks to verify window-level alignment and score provenance.
                ising_rank, _ = ranks_for(ising_params, windows, offset, 'joint4')
                if not np.array_equal(ising_rank, old[panel+'_new']):
                    raise ValueError('Ising historical rank mismatch')
                arrays.update({panel+'_vector': rank, panel+'_ising': ising_rank, panel+'_docs': docs})
                article_diff = np.array([(1/ising_rank[docs == d]-1/rank[docs == d]).mean() for d in np.unique(docs)])
                results[panel] = {'ising_minus_vector_article_mrr': float(article_diff.mean()),
                    'vector_mrr': float(np.mean(1/rank)), 'vector_recall10': float(np.mean(rank <= 10)),
                    'score_parity_max_error': error}
                mlflow.log_metric(panel+'.ising_minus_vector', float(article_diff.mean()))
        wiki = data['validation_windows']
        noise = data['validation_noise']
        for label, model, arm in (('vector', params, 'vector10'), ('ising', ising_params, 'joint4')):
            score = jax.jit(lambda p, w: score_windows(p, w, offset, arm))
            positive, negative = canonical_scores(model, score, wiki, noise)
            outcome = ((positive[:, None] > negative) + .5*(positive[:, None] == negative)).mean(axis=1)
            docs = data['validation_document']
            values = np.array([outcome[docs == d].mean() for d in np.unique(docs)])
            arrays['wiki_auc_'+label] = values
            results['wiki'][label+'_macro_auc'] = float(values.mean())
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
            np.savez_compressed(Path(tmp)/'checkpoint.npz', **params)
            np.savez_compressed(Path(tmp)/'comparison.npz', **arrays)
            mlflow.log_artifacts(tmp, 'results')
        log_contract(mlflow, RunContract('classical', 'cpu_gpu_simulator', 'vector10_full_softmax_log_count',
            'tied vector center dot mean context plus log count', '10-dimensional tied vector', 'not applicable',
            'exact full vocabulary cross-entropy, Adam', seed, sha_array(train),
            'WikiText train and WikiText/EWT development', 'tied vector baseline',
            'development panels reused; matched updates not matched runtime; no physical hardware'),
            {'dataset_run_id': DATA, 'dataset_hashes': meta['hashes'], 'ising_run': ISING[seed],
             'ewt': ewt_meta, 'results': results, 'training_seconds': seconds,
             'checkpoint_hashes': {k: sha_array(v) for k,v in params.items()}})
        mlflow.log_metric('training_seconds', seconds)
        print('FINISHED', run.info.run_id, seed, json.dumps(results), flush=True)


if __name__ == '__main__':
    selected_seed = int(sys.argv[1])
    if selected_seed not in ISING:
        raise ValueError('seed outside protocol')
    main(selected_seed)
