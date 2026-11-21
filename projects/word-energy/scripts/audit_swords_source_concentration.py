"""Posthoc label-blind source grouping for the frozen SWORDS coverage rule."""
from collections import Counter, defaultdict
from pathlib import Path
import gzip
import json
import re
import sys

import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_swords_input_coverage import DATA, SOURCE_SHA
from scripts.build_wikitext2_joint_context_data import load
from word_energy.tracking import sha256_file, tracking_uri
from word_energy.wic_context_probe import extract

source = ROOT/'data/swords/swords-v1.1_dev.json.gz'
if sha256_file(source) != SOURCE_SHA:
    raise ValueError('source hash mismatch')
_, metadata = load(DATA)
vocabulary = {word: i for i,word in enumerate(metadata['vocabulary'])}
with gzip.open(source, 'rt', encoding='utf8') as f:
    data = json.load(f)
data.pop('substitute_labels', None)
candidates = defaultdict(set)
for candidate in data['substitutes'].values():
    word = candidate['substitute'].lower()
    if re.fullmatch(r'[a-z]+', word) and word in vocabulary:
        candidates[candidate['target_id']].add(word)
counts, unknown = Counter(), 0
for target_id, target in data['targets'].items():
    surface = target['target']
    if not re.fullmatch(r'[A-Za-z]+', surface) or surface.lower() not in vocabulary:
        continue
    context = data['contexts'][target['context_id']]
    text = context['context']
    start, end = target['offset'], target['offset']+len(surface)
    if text[start:end].casefold() != surface.casefold() or not extract(text, start, end, vocabulary):
        continue
    if len(candidates[target_id] - {surface.lower()}) < 2:
        continue
    ids = context['extra'].get('coinco_ids', [])
    files = {item.split(':', 1)[0] for item in ids}
    if len(files) != 1:
        unknown += 1
    else:
        counts[next(iter(files))] += 1
output = {'eligible_targets': sum(counts.values())+unknown, 'source_files': len(counts),
          'largest_source_target_count': max(counts.values()), 'unresolved_source_targets': unknown,
          'source_target_counts': dict(counts), 'source_ids_interpretation': 'CoInCo context extra coinco_ids prefix before colon',
          'limitations': 'posthoc label-blind source clustering, metadata not independently validated against original corpus'}
mlflow.set_tracking_uri(tracking_uri())
experiment = mlflow.get_experiment_by_name('word-energy-natural-context-evaluation')
with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='swords-development-source-concentration') as run:
    mlflow.log_dict(output, 'source_concentration.json')
    mlflow.log_artifact(str(Path(__file__)), 'source_snapshot')
    print('RUN', run.info.run_id, json.dumps(output), flush=True)
