"""Paired joint/additive readout on the same frozen joint4 checkpoints."""
from pathlib import Path
import argparse
import json
import sys

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.build_wikitext2_joint_context_data import load
from word_energy.tracking import RunContract, log_contract, tracking_uri, sha256_file
from word_energy.wikitext_four_spin import functions, SEEDS
from word_energy.wikitext_joint_evaluation import canonical_scores
from word_energy.text8_lexical_models import pair_metrics
from word_energy.text8_development import sha_array

DATA = 'f1b7db5231ed4a5788b566e388f8e731'
SUMMARY = 'c05a1d631d814a9db373876c83d7d413'


def audit(trained_arm='joint4'):
    client = MlflowClient()
    arrays, meta = load(DATA)
    summary = json.loads((artifact_dir(client, SUMMARY) / 'config.json').read_text('utf8'))
    assert summary['dataset_hashes'] == meta['hashes']
    windows = arrays['validation_windows']
    noise = arrays['validation_noise']
    docs = arrays['validation_document']
    prior = .25 * np.log(arrays['counts'].astype(float))
    doc_ids = np.unique(docs)
    scorers = {arm: functions(arm, prior)[0] for arm in ('joint4', 'pair_add4')}
    rows, differences = [], []
    for seed in SEEDS:
        record = summary['results'][f'{trained_arm}:{seed}']
        directory = artifact_dir(client, record['run_id'])
        cfg = json.loads((directory / 'config.json').read_text('utf8'))
        assert cfg['dataset_run_id'] == DATA and cfg['offset_sha256'] == sha_array(prior)
        assert sha256_file(ROOT/'word_energy/wikitext_four_spin.py') == sha256_file(directory/'source_snapshot/wikitext_four_spin.py')
        with np.load(directory/'checkpoints/epoch3.npz', allow_pickle=False) as checkpoint:
            params = dict(checkpoint)
        assert {k: sha_array(v) for k,v in params.items()} == cfg['checkpoint_hashes']['epoch3']
        outcomes, values = {}, {}
        for arm, score in scorers.items():
            pos, neg = canonical_scores(params, score, windows, noise)
            outcomes[arm] = (pos[:, None] > neg) + .5 * (pos[:, None] == neg)
            scores = pair_metrics(pos, neg, arrays['counts'][windows[:, 0]])
            by_doc = [float(outcomes[arm][docs == d].mean()) for d in doc_ids]
            values[arm] = dict(macro=float(np.mean(by_doc)), mrr=scores['noise_mrr_at_21'], by_doc=by_doc)
        assert abs(values[trained_arm]['macro'] - record['metrics']['document_macro_auc']) < 1e-12
        delta = outcomes['joint4'] - outcomes['pair_add4']
        diffs = np.subtract(values['joint4']['by_doc'], values['pair_add4']['by_doc'])
        differences.append(diffs)
        rows.append(dict(seed=seed, source=record['run_id'], scores=values, gains=int((delta>0).sum()),
                         losses=int((delta<0).sum()), unchanged=int((delta==0).sum()), macro_delta=float(np.mean(diffs))))
    per_doc = np.mean(differences, axis=0)
    indices = np.random.default_rng(23140).integers(len(doc_ids), size=(1000,len(doc_ids)))
    ci = np.percentile(per_doc[indices].mean(axis=1), [2.5,97.5]).tolist()
    return dict(trained_arm=trained_arm, rows=rows, data=DATA, source_summary=SUMMARY, hashes=meta['hashes'], articles=doc_ids.tolist(),
                windows=len(windows), comparisons_per_seed=noise.size, primary=float(per_doc.mean()), ci=ci,
                advance=bool(ci[0]>0 and all(row['macro_delta']>0 for row in rows)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained-arm', choices=('joint4', 'pair_add4'), default='joint4')
    trained_arm = parser.parse_args().trained_arm
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment('word-energy-wikitext2-four-spin-readout')
    with mlflow.start_run(run_name=f'same-checkpoint-readout-{trained_arm}') as run:
        for path in (Path(__file__), ROOT/'notes/FOUR_SPIN_READOUT_PROTOCOL.md'):
            mlflow.log_artifact(str(path),'source_snapshot')
        out = audit(trained_arm)
        contract = RunContract('classical','cpu_gpu_simulator','four_spin_paired_readout',
            'higher normalized log score plus center prior','12 binary spins','beta=1',
            'exact enumeration by factorization',23140,out['hashes']['validation_windows'],
            'reused WikiText-2 validation','3 K4, 8 cross edges','conditional on fixed checkpoints and reused validation')
        log_contract(mlflow,contract,out)
        mlflow.log_metric('macro_auc_difference',out['primary'])
        print(run.info.run_id, out['primary'], out['ci'], out['advance'])
        for row in out['rows']:
            print(row['seed'], row['macro_delta'], row['gains'], row['losses'], row['unchanged'],
                  [(k,v['macro'],v['mrr']) for k,v in row['scores'].items()])
