"""Train-only WikiText vocabulary size versus label-blind input coverage."""
from collections import Counter, defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import re
import sys

import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_swords_input_coverage import DATA, SOURCE_SHA
from scripts.build_wikitext2_document_data import corpus_data, parse
from scripts.build_wikitext2_joint_context_data import load
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri
from word_energy.wic_context_probe import extract


def main():
    source = ROOT/'data/swords/swords-v1.1_dev.json.gz'
    if sha256_file(source) != SOURCE_SHA:
        raise ValueError('SWORDS development source hash differs')
    _, metadata = load(DATA)
    train_words, train_docs = corpus_data(parse('train'))
    valid_words, valid_docs = corpus_data(parse('validation'))
    frequencies = Counter(word for word in train_words if isinstance(word, str))
    ordered = sorted((word for word, count in frequencies.items() if count >= 5),
                     key=lambda word: (-frequencies[word], word))
    if ordered[:2000] != metadata['vocabulary']:
        raise ValueError('original vocabulary mismatch')
    with gzip.open(source, 'rt', encoding='utf8') as f:
        data = json.load(f)
    data.pop('substitute_labels', None)
    by_target = defaultdict(set)
    for candidate in data['substitutes'].values():
        by_target[candidate['target_id']].add(candidate['substitute'].lower())
    label_free = []
    for target_id, target in data['targets'].items():
        surface = target['target']
        if not re.fullmatch(r'[A-Za-z]+', surface):
            continue
        context = data['contexts'][target['context_id']]['context']
        start = target['offset']
        end = start+len(surface)
        if context[start:end].casefold() != surface.casefold():
            continue
        lemma = target['extra']['coinco_lemma'].lower()
        words = by_target[target_id] - {surface.lower(), lemma}
        label_free.append((surface.lower(), context, start, end, words))
    sizes = (2000, 5000, 10000, 20000, len(ordered))
    outputs = []
    for requested in sizes:
        words = ordered[:requested]
        vocab = {word: index for index, word in enumerate(words)}
        train_in = sum(word in vocab for word in train_words if isinstance(word, str))
        val_in = sum(word in vocab for word in valid_words if isinstance(word, str))
        counts = Counter()
        for surface, context, start, end, candidates in label_free:
            ascii_candidates = {word for word in candidates if re.fullmatch(r'[a-z]+', word)}
            counts['candidate_distinct_all'] += len(candidates)
            counts['candidate_distinct_ascii'] += len(ascii_candidates)
            counts['candidate_distinct_in_vocab'] += len(ascii_candidates.intersection(vocab))
            if surface not in vocab:
                continue
            counts['target_in_vocab'] += 1
            if not extract(context, start, end, vocab):
                continue
            counts['target_and_context'] += 1
            available = len(ascii_candidates.intersection(vocab))
            for threshold in (1, 2, 5, 10):
                if available >= threshold:
                    counts[f'target_context_and_{threshold}_candidates'] += 1
        token_hash = hashlib.sha256('\n'.join(words).encode('utf8')).hexdigest()
        outputs.append({'requested_size': requested, 'actual_size': len(words), 'vocabulary_sha256': token_hash,
                        'train_token_coverage': train_in/int((train_docs >= 0).sum()),
                        'validation_token_coverage': val_in/int((valid_docs >= 0).sum()),
                        'counts': dict(counts), 'ising_word_parameters': 10*len(words)+4,
                        'vector10_parameters': 10*len(words),
                        'candidate_scores_for_600k_full_softmax_windows': 600000*len(words)})
    result = {'train_source': metadata['source_sha256'], 'swords_source_sha256': SOURCE_SHA,
              'prior_data_run': DATA, 'swords_targets_total': len(data['targets']),
              'swords_exact_single_token_targets': len(label_free), 'eligible_train_types_count_ge_5': len(ordered),
              'sizes': outputs, 'limitations': 'label-blind input coverage only; train-only words; development data reused; no model trained, no chip'}
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment('word-energy-natural-context-evaluation')
    with mlflow.start_run(run_name='train-only-vocabulary-expansion-coverage') as run:
        log_contract(mlflow, RunContract('classical', 'cpu_gpu_simulator', 'train_vocabulary_coverage',
            'no model scored', 'ASCII word types', 'not applicable', 'deterministic frequency rank and coverage audit',
            0, SOURCE_SHA, 'WikiText train/validation and SWORDS development inputs only', 'no model graph', result['limitations']), result)
        mlflow.log_dict(result, 'coverage.json')
        for path in (Path(__file__), ROOT/'notes/VOCABULARY_COVERAGE_AUDIT_PROTOCOL.md'):
            mlflow.log_artifact(str(path), 'source_snapshot')
        print('RUN', run.info.run_id, json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
