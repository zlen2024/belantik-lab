"""Frozen exploratory SWORDS candidate-ranking evaluation on development data."""
from collections import Counter, defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import re
import sys
import time

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_swords_input_coverage import DATA, SOURCE_SHA
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.build_wikitext2_joint_context_data import load
from scripts.run_full_softmax_matched_vector import ISING
from scripts.evaluate_full_softmax_wic import VECTORS
from word_energy.contextual_substitution import (ising_candidates, ising_context_only_candidates,
    ising_kernel_candidates, vector_candidates)
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri
from word_energy.wic_context_probe import extract
from word_energy.wikitext_four_spin_wic import star, unary


def build_manifest(data, vocabulary):
    candidates = defaultdict(lambda: defaultdict(list))
    for sid, candidate in data['substitutes'].items():
        word = candidate['substitute'].lower()
        if re.fullmatch(r'[a-z]+', word) and word in vocabulary:
            candidates[candidate['target_id']][word].append(sid)
    manifest, reasons = [], Counter()
    for target_id, target in data['targets'].items():
        surface = target['target']
        if not re.fullmatch(r'[A-Za-z]+', surface) or surface.lower() not in vocabulary:
            reasons['target_oov_or_not_single_token'] += 1
            continue
        context = data['contexts'][target['context_id']]
        text = context['context']
        start, end = target['offset'], target['offset']+len(surface)
        if text[start:end].casefold() != surface.casefold():
            reasons['offset_mismatch'] += 1
            continue
        contexts = extract(text, start, end, vocabulary)
        if not contexts:
            reasons['no_context'] += 1
            continue
        source_ids = {source.split(':', 1)[0] for source in context['extra'].get('coinco_ids', [])}
        if len(source_ids) != 1:
            reasons['ambiguous_source'] += 1
            continue
        lemma = target['extra']['coinco_lemma'].lower()
        words = sorted(set(candidates[target_id]) - {surface.lower(), lemma})
        if len(words) < 2:
            reasons['fewer_than_two_candidates'] += 1
            continue
        manifest.append({'target_id': target_id, 'target': vocabulary[surface.lower()],
                         'context': contexts, 'source': next(iter(source_ids)),
                         'words': words, 'candidate_ids': [vocabulary[word] for word in words],
                         'substitute_ids': [sorted(candidates[target_id][word]) for word in words]})
    return sorted(manifest, key=lambda row: row['target_id']), dict(reasons)


def load_checkpoint(client, run_id, metadata):
    root = artifact_dir(client, run_id)
    config = json.loads((root/'config.json').read_text(encoding='utf8'))
    if config['dataset_hashes'] != metadata['hashes']:
        raise ValueError('training dataset mismatch')
    with np.load(root/'results/checkpoint.npz', allow_pickle=False) as archive:
        params = dict(archive)
    if {key: sha_array(value) for key, value in params.items()} != config['checkpoint_hashes']:
        raise ValueError('checkpoint hash mismatch')
    return params


def pair_auc(scores, labels):
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return None, None
    comparisons = (pos[:, None] > neg) + .5*(pos[:, None] == neg)
    return float(comparisons.mean()), float((pos[:, None] == neg).mean())


def main():
    source = ROOT/'data/swords/swords-v1.1_dev.json.gz'
    if sha256_file(source) != SOURCE_SHA:
        raise ValueError('SWORDS source hash mismatch')
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    arrays, meta = load(DATA)
    vocabulary = {word: i for i,word in enumerate(meta['vocabulary'])}
    with gzip.open(source, 'rt', encoding='utf8') as f:
        data = json.load(f)
    if not data['substitutes_lemmatized']:
        raise ValueError('unexpected candidate format')
    manifest, exclusions = build_manifest(data, vocabulary)
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('utf8')).hexdigest()
    experiment = mlflow.set_experiment('word-energy-natural-context-evaluation')
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='swords-external-candidates-development') as run:
        # Persist eligible IDs and source snapshot before any use of judgments.
        mlflow.log_dict({'manifest_sha256': digest, 'items': manifest, 'exclusions': exclusions}, 'input_manifest.json')
        for path in (Path(__file__), ROOT/'notes/SWORDS_EXTERNAL_CANDIDATE_PROBE_PROTOCOL.md',
                     ROOT/'word_energy/contextual_substitution.py', ROOT/'word_energy/wikitext_four_spin_wic.py'):
            mlflow.log_artifact(str(path), 'source_snapshot')
        models = {seed: {'ising': load_checkpoint(client, ISING[seed], meta),
                         'vector': load_checkpoint(client, VECTORS[seed], meta)} for seed in (17, 29, 43)}
        names = ('ising_affinity', 'ising_target', 'ising_context', 'ising_kernel',
                 'vector_contextual', 'vector_target', 'vector_context', 'vector_dot', 'frequency')
        scored = []
        start_time = time.monotonic()
        for row in manifest:
            target, context = row['target'], row['context']
            ids = np.asarray(row['candidate_ids'], dtype=np.int64)
            item = {'target_id': row['target_id'], 'source': row['source'], 'scores': {}, 'entropy': {}}
            for seed, model in models.items():
                ising, vector = model['ising'], model['vector']
                q = unary(ising)
                target_posterior = star(ising, target, context)
                candidate_entropy = -np.sum(q[ids]*np.log(q[ids]), axis=1)
                item['entropy'][str(seed)] = {'candidate_mean': float(candidate_entropy.mean()),
                                              'target_posterior': float(-np.sum(target_posterior*np.log(target_posterior)))}
                item['scores'][str(seed)] = {
                    'ising_affinity': ising_candidates(ising, target, context, ids).tolist(),
                    'ising_target': ising_candidates(ising, target, context, ids, use_context=False).tolist(),
                    'ising_context': ising_context_only_candidates(ising, context, ids).tolist(),
                    'ising_kernel': ising_kernel_candidates(ising, target, context, ids).tolist(),
                    'vector_contextual': vector_candidates(vector, target, context, ids).tolist(),
                    'vector_target': vector_candidates(vector, target, context, ids, mode='target').tolist(),
                    'vector_context': vector_candidates(vector, target, context, ids, mode='context').tolist(),
                    'vector_dot': vector_candidates(vector, target, context, ids, similarity='dot').tolist(),
                    'frequency': np.log(arrays['counts'][ids].astype(float)).tolist()}
                if any(not np.isfinite(item['scores'][str(seed)][name]).all() for name in names):
                    raise FloatingPointError('nonfinite candidate score')
            scored.append(item)
        scoring_seconds = time.monotonic()-start_time
        mlflow.log_dict(scored, 'label_blind_predictions.json')
        # Judgments are processed only after the frozen manifest and all scores exist.
        results, no_contrast, no_votes, votes_by_target = [], 0, 0, {}
        for row, item in zip(manifest, scored):
            labels, mask, fractions, vote_counts = [], [], [], []
            for sid_list in row['substitute_ids']:
                all_votes = [vote for sid in sid_list for vote in data['substitute_labels'][sid]]
                votes = [vote for vote in all_votes if vote != 'UNSURE']
                if any(vote not in ('TRUE', 'FALSE') for vote in votes):
                    raise ValueError('unexpected judgment label')
                vote_counts.append({'true': votes.count('TRUE'), 'false': votes.count('FALSE'),
                                    'unsure': all_votes.count('UNSURE'), 'total': len(all_votes)})
                if not votes:
                    no_votes += 1
                    labels.append(False)
                    mask.append(False)
                    fractions.append(None)
                else:
                    ratio = votes.count('TRUE')/len(votes)
                    labels.append(ratio > .5)
                    mask.append(True)
                    fractions.append(ratio)
            votes_by_target[row['target_id']] = {'vote_counts': vote_counts, 'true_fraction': fractions}
            label = np.asarray(labels, dtype=bool)[mask]
            if not len(label) or label.all() or not label.any():
                no_contrast += 1
                continue
            row_result = {'target_id': row['target_id'], 'source': row['source'],
                          'true_fraction': fractions, 'vote_counts': vote_counts, 'seed_metrics': {}, 'entropy': item['entropy']}
            for seed in (17, 29, 43):
                scores = item['scores'][str(seed)]
                metrics = {}
                for name in names:
                    values = np.asarray(scores[name])[mask]
                    value, ties = pair_auc(values, label)
                    metrics[name] = {'auc': value, 'tie_fraction': ties, 'score_spread': float(np.ptp(values))}
                original = np.asarray(scores['ising_target'])[mask]
                contextual = np.asarray(scores['ising_affinity'])[mask]
                upper = np.triu_indices(len(original), 1)
                metrics['ising_context_rank_change_fraction'] = float(np.mean(
                    np.sign(original[:, None]-original[None, :])[upper] !=
                    np.sign(contextual[:, None]-contextual[None, :])[upper]))
                row_result['seed_metrics'][str(seed)] = metrics
            results.append(row_result)
        if not results:
            raise ValueError('no informative targets after votes')
        summary = {}
        for name in names:
            means = [float(np.mean([r['seed_metrics'][str(seed)][name]['auc'] for r in results])) for seed in (17, 29, 43)]
            ties = [float(np.mean([r['seed_metrics'][str(seed)][name]['tie_fraction'] for r in results])) for seed in (17, 29, 43)]
            spreads = [float(np.mean([r['seed_metrics'][str(seed)][name]['score_spread'] for r in results])) for seed in (17, 29, 43)]
            summary[name] = {'per_seed': means, 'mean': float(np.mean(means)), 'tie_fraction_per_seed': ties, 'score_spread_per_seed': spreads}
        groups = sorted({r['source'] for r in results})
        group_indices = {group: np.flatnonzero([r['source'] == group for r in results]) for group in groups}
        rng = np.random.default_rng(23140)
        draws = rng.integers(len(groups), size=(1000, len(groups)))
        effects = {}
        for comparator in ('vector_contextual', 'ising_target'):
            deltas = np.asarray([[r['seed_metrics'][str(seed)]['ising_affinity']['auc'] -
                                  r['seed_metrics'][str(seed)][comparator]['auc'] for r in results] for seed in (17, 29, 43)])
            mean_by_target = deltas.mean(axis=0)
            bootstrap = []
            for draw in draws:
                ids = np.concatenate([group_indices[groups[i]] for i in draw])
                bootstrap.append(float(mean_by_target[ids].mean()))
            without = np.array([r['source'] != 'Nathans_Bylichka.txt' for r in results])
            effects[comparator] = {'mean_difference': float(mean_by_target.mean()),
                'per_seed': deltas.mean(axis=1).tolist(), 'ci95': np.percentile(bootstrap, [2.5, 97.5]).tolist(),
                'valid_draws': len(bootstrap), 'without_dominant_source': float(mean_by_target[without].mean()) if without.any() else None,
                'without_dominant_count': int(without.sum())}
        reduced = [r for r in results if r['source'] != 'Nathans_Bylichka.txt']
        reduced_groups = sorted({r['source'] for r in reduced})
        sensitivity = {'targets': len(reduced), 'source_counts': dict(Counter(r['source'] for r in reduced)),
                       'summary': {}, 'effects': {}}
        if reduced:
            reduced_indices = {group: np.flatnonzero([r['source'] == group for r in reduced]) for group in reduced_groups}
            reduced_draws = np.random.default_rng(23140).integers(len(reduced_groups), size=(1000, len(reduced_groups)))
            for name in names:
                per_seed = [float(np.mean([r['seed_metrics'][str(seed)][name]['auc'] for r in reduced])) for seed in (17, 29, 43)]
                ties = [float(np.mean([r['seed_metrics'][str(seed)][name]['tie_fraction'] for r in reduced])) for seed in (17, 29, 43)]
                spreads = [float(np.mean([r['seed_metrics'][str(seed)][name]['score_spread'] for r in reduced])) for seed in (17, 29, 43)]
                sensitivity['summary'][name] = {'per_seed': per_seed, 'mean': float(np.mean(per_seed)),
                                                'tie_fraction_per_seed': ties, 'score_spread_per_seed': spreads}
            for comparator in ('vector_contextual', 'ising_target'):
                differences = np.asarray([[r['seed_metrics'][str(seed)]['ising_affinity']['auc'] -
                                           r['seed_metrics'][str(seed)][comparator]['auc'] for r in reduced]
                                          for seed in (17, 29, 43)])
                pooled = differences.mean(axis=0)
                boot = [float(pooled[np.concatenate([reduced_indices[reduced_groups[i]] for i in draw])].mean())
                        for draw in reduced_draws]
                sensitivity['effects'][comparator] = {'mean_difference': float(pooled.mean()),
                    'per_seed': differences.mean(axis=1).tolist(), 'ci95': np.percentile(boot, [2.5, 97.5]).tolist(),
                    'valid_draws': len(boot)}
        diagnostics = {'mean_candidate_entropy': float(np.mean([r['entropy'][str(seed)]['candidate_mean'] for r in results for seed in (17,29,43)])),
                       'mean_target_posterior_entropy': float(np.mean([r['entropy'][str(seed)]['target_posterior'] for r in results for seed in (17,29,43)])),
                       'mean_rank_change_fraction': float(np.mean([r['seed_metrics'][str(seed)]['ising_context_rank_change_fraction'] for r in results for seed in (17,29,43)]))}
        output = {'manifest_sha256': digest, 'total_targets': len(data['targets']),
                  'eligible_targets': len(manifest), 'eligible_candidate_ids': sum(len(r['candidate_ids']) for r in manifest),
                  'exclusions': exclusions, 'informative_targets': len(results),
                  'no_contrast_targets': no_contrast, 'candidate_entries_no_votes': no_votes,
                  'sources': len(groups), 'source_counts': dict(Counter(r['source'] for r in results)),
                  'summary': summary, 'effects': effects, 'without_dominant_source': sensitivity,
                  'diagnostics': diagnostics, 'scoring_seconds': scoring_seconds,
                  'limitations': 'SWORDS development, filtered candidate vocabulary, repeated model selection, external-candidate host scores, no Gaussian or chip'}
        log_contract(mlflow, RunContract('classical', 'cpu_gpu_simulator', 'swords_external_candidate_ranking',
            'exact posterior/candidate affinity, kernel and vector controls', '4 spins or tied 10D vector', 'Ising beta=1',
            'exact state posterior, host-side candidate readout', 23140, SOURCE_SHA,
            'SWORDS development, official test unopened', '4*(1+m) active spins, m=1..4', output['limitations']), output)
        mlflow.log_dict(output, 'summary.json')
        mlflow.log_dict(results, 'target_results.json')
        mlflow.log_dict(votes_by_target, 'candidate_vote_counts.json')
        for name in names:
            mlflow.log_metric(name+'.target_macro_auc', summary[name]['mean'])
        print('RUN', run.info.run_id, json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
