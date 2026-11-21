"""Measure candidate vocabulary exposure under a reproducible window sampler."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import time

import mlflow
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_wikitext2_document_data import corpus_data, parse, sample
from scripts.build_wikitext2_joint_context_data import load, windows
from scripts.run_full_softmax_prior import DATA
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract, log_contract, tracking_uri


def exposure(values, names, begin, end):
    center = np.bincount(values[:, 0], minlength=len(names))[begin:end]
    occurrences = np.bincount(values.ravel(), minlength=len(names))[begin:end]
    def describe(counts):
        return {'zero': int(np.sum(counts == 0)), 'one': int(np.sum(counts == 1)),
                'five_or_more': int(np.sum(counts >= 5)), 'ten_or_more': int(np.sum(counts >= 10)),
                'quantiles_0_25_50_75_100': np.percentile(counts, (0, 25, 50, 75, 100)).tolist(),
                'sum': int(counts.sum())}
    return {'rank_begin': begin+1, 'rank_end': end, 'types': end-begin,
            'sampled_centers': describe(center), 'sampled_all_positions': describe(occurrences)}


def main():
    start = time.monotonic()
    original, metadata = load(DATA)
    docs = parse('train')
    raw, _ = corpus_data(docs)
    frequencies = Counter(word for word in raw if isinstance(word, str))
    names = sorted((word for word, count in frequencies.items() if count >= 5),
                   key=lambda word: (-frequencies[word], word))[:10000]
    if names[:2000] != metadata['vocabulary'] or len(names) != 10000:
        raise ValueError('vocabulary mismatch')
    ids, document = corpus_data(docs, {word: index for index, word in enumerate(names)})
    ids = np.asarray(ids, dtype=np.int32)
    pairs, positions, groups = sample(ids, document, 200000, np.random.default_rng(22101))
    selected, pos, selected_docs, _, eligible_pairs = windows(ids, document, pairs, positions, groups, 200000, 23101)
    if len(selected) != 200000:
        raise ValueError('window count mismatch')
    candidate_arrays = {'train_windows': selected, 'train_positions': pos, 'train_document': selected_docs}
    vocabulary_hash = hashlib.sha256('\n'.join(names).encode('utf8')).hexdigest()
    def unique_windows(p):
        canonical = np.column_stack((p[:, 0], np.sort(p[:, 1:], axis=1)))
        return len(np.unique(canonical, axis=0))
    output = {'dataset_run': DATA, 'source_sha256': metadata['source_sha256'], 'vocabulary_sha256': vocabulary_hash,
              'original_vocabulary_size': 2000, 'candidate_vocabulary_size': len(names),
              'original_window_count': len(original['train_windows']), 'candidate_window_count': len(selected),
              'eligible_pair_rows_in_initial_sample': eligible_pairs,
              'original_unique_occurrence_windows': unique_windows(original['train_positions']),
              'candidate_unique_occurrence_windows': unique_windows(pos),
              'original_rank_1_2000': exposure(original['train_windows'], metadata['vocabulary'], 0, 2000),
              'candidate_rank_1_2000': exposure(selected, names, 0, 2000),
              'candidate_rank_2001_5000': exposure(selected, names, 2000, 5000),
              'candidate_rank_5001_10000': exposure(selected, names, 5000, 10000),
              'candidate_array_hashes': {k: sha_array(v) for k,v in candidate_arrays.items()},
              'elapsed_seconds': time.monotonic()-start,
              'limitations': 'prospective sampled corpus windows, no model training, changed sampling eligibility, softmax still scores all classes'}
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment('word-energy-natural-context-evaluation')
    with mlflow.start_run(run_name='ten-thousand-vocabulary-window-exposure') as run:
        log_contract(mlflow, RunContract('classical', 'cpu_gpu_simulator', 'vocabulary_window_exposure',
            'no model scored', 'integer word IDs', 'not applicable', 'fixed pair/window sampling',
            23101, vocabulary_hash, 'WikiText train only; no labels or reserved test', 'no model graph', output['limitations']), output)
        mlflow.log_dict(output, 'exposure.json')
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
            np.savez_compressed(Path(tmp)/'candidate_windows.npz', **candidate_arrays)
            (Path(tmp)/'vocabulary.json').write_text(json.dumps(names), encoding='utf8')
            mlflow.log_artifacts(tmp, 'candidate_data')
        for path in (Path(__file__), ROOT/'notes/VOCABULARY_EXPOSURE_AUDIT_PROTOCOL.md',
                     ROOT/'scripts/build_wikitext2_joint_context_data.py', ROOT/'scripts/build_wikitext2_document_data.py'):
            mlflow.log_artifact(str(path), 'source_snapshot')
        print('RUN', run.info.run_id, json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
