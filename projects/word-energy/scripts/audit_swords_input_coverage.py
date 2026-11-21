"""Label-blind SWORDS development input and vocabulary coverage."""
from collections import Counter, defaultdict
from pathlib import Path
import gzip
import json
import re
import sys

import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_wikitext2_document_data import norm, parse
from scripts.build_wikitext2_joint_context_data import load
from word_energy.tracking import RunContract, log_contract, tracking_uri, sha256_file
from word_energy.wic_context_probe import extract

SOURCE_SHA = 'bc3570786455a05f11fcb4fc7f54e117642d5fb5e98f19355028c5c7ee00c954'
DATA = 'f1b7db5231ed4a5788b566e388f8e731'


def main():
    source = ROOT/'data/swords/swords-v1.1_dev.json.gz'
    if sha256_file(source) != SOURCE_SHA:
        raise ValueError('SWORDS development source hash differs')
    _, metadata = load(DATA)
    vocabulary = {word: i for i,word in enumerate(metadata['vocabulary'])}
    with gzip.open(source, 'rt', encoding='utf8') as f:
        data = json.load(f)
    # Ignore judgments completely. Candidate availability is not label quality.
    data.pop('substitute_labels', None)
    candidates = defaultdict(set)
    for candidate in data['substitutes'].values():
        text = candidate['substitute'].lower()
        if re.fullmatch(r'[a-z]+', text) and text in vocabulary:
            candidates[candidate['target_id']].add(text)
    wiki_rows = [norm(row) for doc in parse('train')+parse('validation') for row in doc['rows']]
    wiki_joined = '\n'.join(wiki_rows)
    counts = Counter()
    candidate_histogram = Counter()
    for target_id, target in data['targets'].items():
        counts['total_targets'] += 1
        text = data['contexts'][target['context_id']]['context']
        normal = norm(text)
        if len(normal) >= 80 and normal in wiki_joined:
            counts['exact_long_context_overlap'] += 1
        surface = target['target']
        if not re.fullmatch(r'[A-Za-z]+', surface):
            counts['non_single_token_target'] += 1
            continue
        start = target['offset']
        end = start + len(surface)
        if text[start:end].casefold() != surface.casefold():
            counts['target_offset_mismatch'] += 1
            continue
        counts['single_token_exact_span'] += 1
        if surface.lower() not in vocabulary:
            counts['target_oov'] += 1
            continue
        counts['target_in_vocabulary'] += 1
        contexts = extract(text, start, end, vocabulary)
        if contexts is None or not contexts:
            counts['no_eligible_context'] += 1
            continue
        counts['target_and_context_eligible'] += 1
        available = candidates[target_id] - {surface.lower()}
        candidate_histogram[str(min(len(available), 10))] += 1
        for threshold in (1, 2, 5, 10):
            if len(available) >= threshold:
                counts[f'eligible_with_{threshold}_candidate_ids'] += 1
    output = {'source': 'https://github.com/p-lambda/swords', 'source_sha256': SOURCE_SHA,
              'license': 'CC-BY-3.0-US', 'data_run': DATA, 'vocabulary_size': len(vocabulary),
              'counts': dict(counts), 'candidate_count_histogram_capped_at_10': dict(candidate_histogram),
              'limitations': 'development inputs only, no labels or model scores, exact long substring overlap screen only'}
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment('word-energy-natural-context-evaluation')
    with mlflow.start_run(run_name='swords-development-label-blind-coverage') as run:
        log_contract(mlflow, RunContract('classical', 'cpu_gpu_simulator', 'swords_input_coverage',
            'no model scored', 'single-token vocabulary IDs', 'not applicable', 'deterministic coverage audit',
            0, SOURCE_SHA, 'SWORDS development inputs only; test and labels unused', 'no model graph', output['limitations']), output)
        mlflow.log_dict(output, 'coverage.json')
        for path in (Path(__file__), ROOT/'notes/SWORDS_COVERAGE_AUDIT_PROTOCOL.md'):
            mlflow.log_artifact(str(path), 'source_snapshot')
        print('RUN', run.info.run_id, json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
