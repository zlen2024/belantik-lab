"""Audit matched candidate-preference reversals in reused SWORDS development."""
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path
import hashlib
import json
import sys

import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri

SOURCE = '6bfbcadfb039410dbc206f963b50d2bf'


def context_hash(context):
    return hashlib.sha256(json.dumps(context, separators=(',', ':')).encode()).hexdigest()


def summarize(matches):
    targets, contexts, candidate_pairs, sources = set(), set(), set(), set()
    same_source, cross_source = 0, 0
    for match in matches:
        targets.update(match['targets'])
        contexts.update(match['contexts'])
        candidate_pairs.add(tuple(match['candidates']))
        source_sets = [set(row) for row in match['source_sets']]
        sources.update(set().union(*source_sets))
        same_source += bool(set.intersection(*source_sets))
        cross_source += len(set.union(*source_sets)) > 1
    return {'matches': len(matches), 'distinct_target_ids': len(targets), 'distinct_context_bags': len(contexts),
            'distinct_candidate_pairs': len(candidate_pairs), 'source_file_proxies': len(sources),
            'matches_sharing_at_least_one_source': same_source,
            'matches_with_more_than_one_source': cross_source}


def main():
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    root = artifact_dir(client, SOURCE)
    manifest_data = json.loads((root/'input_manifest.json').read_text(encoding='utf8'))
    manifest = manifest_data['items']
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != manifest_data['manifest_sha256']:
        raise ValueError('input manifest hash mismatch')
    scored = json.loads((root/'target_results.json').read_text(encoding='utf8'))
    rows = {row['target_id']: row for row in scored}
    if len(manifest) != 219 or len(rows) != 132:
        raise ValueError('source subset changed')
    cells = defaultdict(lambda: {'signs': set(), 'sources': set(), 'targets': set()})
    pair_count = 0
    for item in manifest:
        if item['target_id'] not in rows:
            continue
        row = rows[item['target_id']]
        if row['source'] != item['source'] or len(row['true_fraction']) != len(item['candidate_ids']):
            raise ValueError('source candidate alignment mismatch')
        entries = [(candidate, fraction > .5) for candidate, fraction in zip(item['candidate_ids'], row['true_fraction']) if fraction is not None]
        context = context_hash(item['context'])
        for left, right in combinations(entries, 2):
            if left[1] == right[1]:
                continue
            lo, hi = sorted((left[0], right[0]))
            sign = (left[0] == lo and left[1]) or (right[0] == lo and right[1])
            cell = cells[(item['target'], context, lo, hi)]
            cell['signs'].add(bool(sign))
            cell['sources'].add(item['source'])
            cell['targets'].add(item['target_id'])
            pair_count += 1
    conflicts = sum(len(value['signs']) != 1 for value in cells.values())
    clean = {key: value for key,value in cells.items() if len(value['signs']) == 1}
    context_groups, target_groups, candidate_groups = defaultdict(list), defaultdict(list), defaultdict(dict)
    for (target, context, lo, hi), value in clean.items():
        sign = next(iter(value['signs']))
        sources = sorted(value['sources'])
        context_groups[(target, lo, hi)].append((context, sign, sources))
        target_groups[(context, lo, hi)].append((target, sign, sources))
        candidate_groups[(lo, hi)].setdefault(target, {})[context] = (sign, sources)
    context_matches = []
    for (target, lo, hi), items in context_groups.items():
        for first, second in combinations(items, 2):
            if first[1] != second[1]:
                context_matches.append({'targets': [target], 'contexts': sorted((first[0], second[0])),
                                        'candidates': [lo, hi], 'source_sets': [first[2], second[2]]})
    target_matches = []
    for (context, lo, hi), items in target_groups.items():
        for first, second in combinations(items, 2):
            if first[1] != second[1]:
                target_matches.append({'targets': sorted((first[0], second[0])), 'contexts': [context],
                                       'candidates': [lo, hi], 'source_sets': [first[2], second[2]]})
    checkerboards = []
    for (lo, hi), targets in candidate_groups.items():
        for first, second in combinations(sorted(targets), 2):
            common = sorted(set(targets[first]) & set(targets[second]))
            for left, right in combinations(common, 2):
                p = [targets[first][left], targets[first][right], targets[second][left], targets[second][right]]
                if p[0][0] == p[3][0] and p[1][0] == p[2][0] and p[0][0] != p[1][0]:
                    checkerboards.append({'targets': [first, second], 'contexts': [left, right],
                                          'candidates': [lo, hi], 'source_sets': [entry[1] for entry in p]})
    output = {'source_run': SOURCE, 'manifest_sha256': digest,
              'source_artifact_sha256': {name: sha256_file(root/name) for name in ('input_manifest.json', 'target_results.json')},
              'informative_targets': len(rows), 'initial_strict_candidate_pairs': pair_count,
              'distinct_cells': len(cells), 'conflicting_cells_excluded': conflicts,
              'context_reversals': summarize(context_matches), 'target_reversals': summarize(target_matches),
              'checkerboards': summarize(checkerboards),
              'limitations': 'posthoc source-filtered development labels, strict exact token-bag matching, no model training, no physical hardware'}
    mlflow.set_experiment('word-energy-natural-context-evaluation')
    with mlflow.start_run(run_name='swords-posthoc-crossed-support-audit') as run:
        log_contract(mlflow, RunContract('classical', 'cpu_gpu_simulator', 'swords_crossed_support_audit',
            'no model scored', 'integer token IDs', 'not applicable', 'deterministic strict preference counting',
            0, digest, 'reused SWORDS development judgments, official test closed', 'no model graph', output['limitations']), output)
        mlflow.log_dict(output, 'audit.json')
        mlflow.log_dict({'context_reversals': context_matches, 'target_reversals': target_matches,
                         'checkerboards': checkerboards}, 'matched_cells.json')
        for path in (Path(__file__), ROOT/'notes/SWORDS_CROSSED_SUPPORT_AUDIT_PROTOCOL.md'):
            mlflow.log_artifact(str(path), 'source_snapshot')
        print('RUN', run.info.run_id, json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
