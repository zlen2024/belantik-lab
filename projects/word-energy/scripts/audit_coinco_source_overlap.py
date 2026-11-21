"""Posthoc source-file overlap of CoInCo and SWORDS development inputs."""
from collections import Counter
from pathlib import Path
import gzip
import json
import sys

import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_swords_input_coverage import SOURCE_SHA
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri

COINCO_SHA = '02180be2bf9e3b528682c659c5b410f3baa130dbc19a29be8755a0aae0d9cc5f'


def load_source(path, expected):
    if sha256_file(path) != expected:
        raise ValueError('source hash mismatch')
    with gzip.open(path, 'rt', encoding='utf8') as f:
        data = json.load(f)
    # Serialized judgments are loaded but never used in this input audit.
    data.pop('substitute_labels', None)
    return data


def main():
    coinco = load_source(ROOT/'data/swords/coinco_dev.json.gz', COINCO_SHA)
    swords = load_source(ROOT/'data/swords/swords-v1.1_dev.json.gz', SOURCE_SHA)
    swords_sources = {value.split(':', 1)[0] for row in swords['contexts'].values()
                      for value in row.get('extra', {}).get('coinco_ids', [])}
    coinco_sources = {}
    for context_id, context in coinco['contexts'].items():
        coinco_sources[context_id] = {item['masc']['document_fn'] for item in context.get('extra', [])
                                      if item.get('masc', {}).get('document_fn')}
    all_sources = set().union(*coinco_sources.values())
    categories = Counter()
    for sources in coinco_sources.values():
        if not sources:
            categories['missing_source_contexts'] += 1
        elif len(sources) > 1:
            categories['multi_source_contexts'] += 1
        elif sources & swords_sources:
            categories['shared_source_contexts'] += 1
        else:
            categories['nonshared_source_contexts'] += 1
    target_categories = Counter()
    distinct_contexts = Counter()
    targets = Counter()
    for context_id, sources in coinco_sources.items():
        if len(sources) == 1 and not sources & swords_sources:
            distinct_contexts[next(iter(sources))] += 1
    for target in coinco['targets'].values():
        sources = coinco_sources[target['context_id']]
        if not sources:
            target_categories['missing_source_targets'] += 1
        elif len(sources) > 1:
            target_categories['multi_source_targets'] += 1
        elif sources & swords_sources:
            target_categories['shared_source_targets'] += 1
        else:
            target_categories['nonshared_source_targets'] += 1
            targets[next(iter(sources))] += 1
    output = {'coinco_sha256': COINCO_SHA, 'swords_sha256': SOURCE_SHA,
              'coinco_contexts': len(coinco['contexts']), 'coinco_targets': len(coinco['targets']),
              'coinco_source_proxies': len(all_sources), 'swords_source_proxies': len(swords_sources),
              'shared_source_proxies': len(all_sources & swords_sources),
              'nonshared_source_proxies': len(all_sources-swords_sources),
              'context_categories': dict(categories), 'target_categories': dict(target_categories),
              'nonshared_source_counts': {source: {'contexts': distinct_contexts[source], 'targets': targets[source]}
                                          for source in sorted(all_sources-swords_sources)},
              'limitations': 'posthoc metadata-only source proxies, no labels or model, nonmatching source does not prove no data leakage'}
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment('word-energy-natural-context-evaluation')
    with mlflow.start_run(run_name='coinco-swords-development-source-overlap') as run:
        log_contract(mlflow, RunContract('classical', 'cpu_gpu_simulator', 'coinco_source_overlap',
            'no model scored', 'source identifiers', 'not applicable', 'deterministic input-metadata comparison',
            0, COINCO_SHA, 'CoInCo and SWORDS development input metadata; tests closed', 'no model graph', output['limitations']), output)
        mlflow.log_dict(output, 'overlap.json')
        for path in (Path(__file__), ROOT/'notes/COINCO_SOURCE_OVERLAP_AUDIT_PROTOCOL.md'):
            mlflow.log_artifact(str(path), 'source_snapshot')
        print('RUN', run.info.run_id, json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
