"""Compare full-vocabulary conditional training with frozen margin checkpoints."""
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
from word_energy.ising.exact_training import _imports
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract, log_contract, tracking_uri
from word_energy.wikitext_four_spin import initial, score_windows, state_table, unary_probabilities

DATA = 'f1b7db5231ed4a5788b566e388f8e731'
REFERENCES = {'wiki': '782220fbb92b4f18ac410c53e0e44ff4',
              'ewt': '6219acde7e58415ba42ba9b067f7061e'}


def make_functions(prior):
    jax, jnp, optax = _imports()
    offset = jnp.asarray(prior)

    def logits(params, windows):
        states = state_table(jnp)
        q = unary_probabilities(params)
        coupling = 1.5 * jax.nn.sigmoid(params['interaction']) / 2
        kernel = jnp.expm1((states * coupling) @ states.T)
        messages = q @ kernel.T
        left, right = messages[windows[:, 1]], messages[windows[:, 2]]
        return jnp.log1p((left + right + left * right) @ q.T) + offset

    def loss(params, windows):
        scores = logits(params, windows)
        return -jnp.mean(jax.nn.log_softmax(scores)[jnp.arange(len(windows)), windows[:, 0]])

    optimizer = optax.chain(optax.clip_by_global_norm(5.), optax.adam(.01))

    @jax.jit
    def step(params, state, windows):
        value, grad = jax.value_and_grad(loss)(params, windows)
        updates, state = optimizer.update(grad, state, params)
        return optax.apply_updates(params, updates), state, value

    return jax.jit(logits), jax.jit(jax.value_and_grad(loss)), step, optimizer


def main(seed):
    jax, jnp, _ = _imports()
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    data, meta = load(DATA)
    prior = .25 * np.log(data['counts'].astype(float))
    params = initial('joint4', seed, len(prior))
    logits, loss_grad, step, optimizer = make_functions(prior)
    probe = data['train_windows'][:8]
    matrix = np.asarray(logits(params, probe))
    for candidates in (probe[:, 0], (probe[:, 0] + 713) % len(prior)):
        w = np.column_stack((candidates, probe[:, 1:]))
        reference = np.asarray(score_windows(params, w, prior, 'joint4'))
        np.testing.assert_allclose(matrix[np.arange(len(w)), candidates], reference, atol=1e-11, rtol=0)
    val, grad = loss_grad(params, probe)
    if not np.isfinite(float(val)) or any(not np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(grad)):
        raise FloatingPointError('nonfinite gradient precheck')
    eps = 1e-5
    plus = dict(params)
    minus = dict(params)
    plus['biases'] = params['biases'].at[int(probe[0, 0]), 0].add(eps)
    minus['biases'] = params['biases'].at[int(probe[0, 0]), 0].add(-eps)
    finite_difference = (float(loss_grad(plus, probe)[0]) - float(loss_grad(minus, probe)[0])) / (2 * eps)
    np.testing.assert_allclose(finite_difference, np.asarray(grad['biases'])[probe[0, 0], 0], atol=1e-7, rtol=1e-4)
    ewt, ewt_docs, ewt_meta = build_ewt(ROOT, meta['vocabulary'])
    panels = {'wiki': (data['validation_windows'], data['validation_document']), 'ewt': (ewt, ewt_docs)}
    mlflow.set_experiment('word-energy-full-softmax-pilot')
    with mlflow.start_run(run_name=f'full-softmax-joint4-seed{seed}') as run:
        mlflow.set_tags({'seed': str(seed), 'dataset_run_id': DATA, 'test_status': 'reserved closed'})
        mlflow.log_params({'seed': seed, 'batch': 256, 'epochs': 3, 'lr': .01,
                           'objective': 'full-vocabulary cross-entropy', 'vocabulary': len(prior)})
        for path in (Path(__file__), ROOT/'notes/FULL_SOFTMAX_PILOT_PROTOCOL.md', ROOT/'word_energy/wikitext_four_spin.py'):
            mlflow.log_artifact(str(path), 'source_snapshot')
        state = optimizer.init(params)
        start = time.monotonic()
        train = data['train_windows']
        for epoch in range(3):
            total = 0.
            for i in range(0, len(train), 256):
                ix = data['orders'][epoch, i:i+256]
                params, state, value = step(params, state, train[ix])
                value = float(value)
                if not np.isfinite(value):
                    raise FloatingPointError('nonfinite training loss')
                total += len(ix) * value
            mlflow.log_metric('train_cross_entropy', total / len(train), step=epoch + 1)
            print(seed, 'epoch', epoch + 1, 'loss', total / len(train), 'seconds', time.monotonic()-start, flush=True)
        seconds = time.monotonic() - start
        params = {k: np.asarray(v) for k, v in params.items()}
        arrays, results = {}, {}
        summary = json.loads((artifact_dir(client, 'c05a1d631d814a9db373876c83d7d413')/'config.json').read_text('utf8'))
        source = summary['results'][f'joint4:{seed}']['run_id']
        with np.load(artifact_dir(client, source)/'checkpoints/epoch3.npz', allow_pickle=False) as saved:
            margin_params = dict(saved)
        for panel, (windows, docs) in panels.items():
            raw, error = ranks_for(params, windows, prior, 'joint4')
            corrected, _ = ranks_for(params, windows, prior + np.log(data['noise_q']), 'joint4')
            margin_raw, _ = ranks_for(margin_params, windows, prior, 'joint4')
            with np.load(artifact_dir(client, REFERENCES[panel])/'ranks/ranks.npz') as baseline:
                if not np.array_equal(docs, baseline['doc']):
                    raise ValueError('reference document mismatch')
                old = baseline[f'joint4_{seed}'].copy()
            article = np.array([(1/raw[docs == d] - 1/old[docs == d]).mean() for d in np.unique(docs)])
            arrays.update({panel+'_raw': raw, panel+'_corrected': corrected, panel+'_baseline': old,
                           panel+'_margin_raw': margin_raw, panel+'_docs': docs, panel+'_article_delta': article})
            results[panel] = {'article_mrr_delta': float(article.mean()), 'mrr': float(np.mean(1/raw)),
                              'recall10': float(np.mean(raw <= 10)), 'score_parity_max_error': error}
            mlflow.log_metric(panel+'.article_mrr_delta', float(article.mean()))
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
            np.savez_compressed(Path(tmp)/'checkpoint.npz', **params)
            np.savez_compressed(Path(tmp)/'ranks.npz', **arrays)
            mlflow.log_artifacts(tmp, 'results')
        log_contract(mlflow, RunContract('classical', 'cpu_gpu_simulator', 'joint4_full_softmax',
            'normalized Ising joint score plus quarter log count', '12 binary spins', 'beta=1',
            'exact factorization, full vocabulary, Adam', seed, sha_array(train),
            'WikiText train and WikiText/EWT development', '3 K4 and 8 spokes',
            'development panels reused; unequal training compute; no physical hardware'),
            {'data': DATA, 'dataset_hashes': meta['hashes'], 'references': REFERENCES,
             'ewt': ewt_meta, 'results': results, 'training_seconds': seconds,
             'margin_source': source, 'candidate_scores': int(len(train)*3*len(prior)),
             'checkpoint_hashes': {k: sha_array(v) for k, v in params.items()}})
        mlflow.log_metric('training_seconds', seconds)
        print('FINISHED', run.info.run_id, seed, json.dumps(results), flush=True)


if __name__ == '__main__':
    seed = int(sys.argv[1])
    if seed not in (17, 29, 43):
        raise ValueError('seed outside protocol')
    main(seed)
