"""Post-hoc, paired document bootstrap and cue-permutation control."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import sys
import zipfile
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.evaluate_semcor_grouped_baselines import ARCHIVE, VOCAB_RUN, collect, key
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri

def main():
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    baseline = json.loads((artifact_dir(client, 'bd4c107ad9654a1288761a207dcd7d98') / 'results.json').read_text())
    meta = json.loads((artifact_dir(client, VOCAB_RUN) / 'config.json').read_text())
    words = meta['vocabulary']
    assert hashlib.sha256(json.dumps(words, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest() == baseline['vocabulary_sha256']
    archive = ROOT / 'data/semcor/semcor.zip'
    assert sha256_file(archive) == ARCHIVE
    vocab = {w: i for i, w in enumerate(words)}
    with zipfile.ZipFile(archive) as z:
        names = [x for x in z.namelist() if '/tagfiles/' in x and x.endswith('.xml')]
        assert len(names) == 352 and len({Path(x).name for x in names}) == 352
        outer = sorted(names, key=lambda n: key('semcor-fit-v1:', n))[:282]
        inner = sorted(outer, key=lambda n: key('semcor-inner-v1:', n))
        assert [key('semcor-inner-v1:', n) for n in inner[:225]] == baseline['inner_fit_document_hashes']
        assert [key('semcor-inner-v1:', n) for n in inner[225:]] == baseline['inner_validation_document_hashes']
        train = collect(z, inner[:225], vocab, set(words[:30]))
        val = collect(z, inner[225:], vocab, set(words[:30]))
    rows, groups = [], defaultdict(list)
    for word, senses in sorted(train.items()):
        qualified = [(s, xs) for s, xs in senses.items() if len(xs) >= 5 and len({x['doc'] for x in xs}) >= 3]
        if len(qualified) < 2:
            continue
        items = dict(sorted(qualified, key=lambda p: (-len(p[1]), p[0]))[:2])
        labels = sorted(items)
        counts = {s: len(items[s]) for s in labels}
        majority = sorted(labels, key=lambda s: (-counts[s], s))[0]
        tokens = {s: Counter(c for x in items[s] for c in x['cues']) for s in labels}
        totals = {s: sum(tokens[s].values()) for s in labels}
        def predict(cues):
            scores = {s: math.log(counts[s] / sum(counts.values())) + sum(math.log((tokens[s][c] + 1) / (totals[s] + len(words))) for c in cues) for s in labels}
            return sorted(labels, key=lambda s: (-scores[s], s))[0]
        for label in labels:
            for example in val.get(word, {}).get(label, []):
                row = {'document': hashlib.sha256(('semcor-doc-v1:' + example['doc']).encode()).hexdigest(), 'target': word, 'sense': label, 'cue_count': len(example['cues']), 'cues': example['cues'], 'majority_correct': int(majority == label), 'cue_correct': int(predict(example['cues']) == label)}
                row['predict'] = predict
                groups[(word, len(example['cues']))].append(len(rows))
                rows.append(row)
    assert len(rows) == baseline['selected_validation_occurrences'] == 1279
    assert sum(r['majority_correct'] for r in rows) == 850 and sum(r['cue_correct'] for r in rows) == 876
    # Bind each closure to its own target; nested-loop late binding is unsafe.
    # The scorer below is reconstructed by target to retain fixed model parameters.
    models = {}
    for word in {r['target'] for r in rows}:
        qualified = [(s, xs) for s, xs in train[word].items() if len(xs) >= 5 and len({x['doc'] for x in xs}) >= 3]
        items = dict(sorted(qualified, key=lambda p: (-len(p[1]), p[0]))[:2]); labels = sorted(items)
        counts = {s: len(items[s]) for s in labels}
        tokens = {s: Counter(c for x in items[s] for c in x['cues']) for s in labels}
        totals = {s: sum(tokens[s].values()) for s in labels}
        models[word] = (labels, counts, tokens, totals)
    def score(word, cues):
        labels, counts, tokens, totals = models[word]
        scores = {s: math.log(counts[s] / sum(counts.values())) + sum(math.log((tokens[s][c] + 1) / (totals[s] + len(words))) for c in cues) for s in labels}
        return sorted(labels, key=lambda s: (-scores[s], s))[0]
    assert all(int(score(r['target'], r['cues']) == r['sense']) == r['cue_correct'] for r in rows)
    rng = np.random.default_rng(20261122)
    docs = sorted({r['document'] for r in rows})
    doc_rows = {d: [r for r in rows if r['document'] == d] for d in docs}
    boot = []
    for _ in range(2000):
        sampled = rng.choice(docs, size=len(docs), replace=True)
        picked = [r for d in sampled for r in doc_rows[d]]
        boot.append(sum(r['cue_correct'] - r['majority_correct'] for r in picked) / len(picked))
    shuffled = []
    for _ in range(200):
        correct = 0
        for indices in groups.values():
            perm = rng.permutation(indices)
            for dest, source in zip(indices, perm):
                r = rows[dest]
                correct += int(score(r['target'], rows[source]['cues']) == r['sense'])
        shuffled.append(correct / len(rows))
    strata = {str(n): {'examples': len(sub), 'majority_accuracy': sum(r['majority_correct'] for r in sub)/len(sub), 'cue_accuracy': sum(r['cue_correct'] for r in sub)/len(sub)} for n in (2, 3, 4) if (sub := [r for r in rows if r['cue_count'] == n])}
    result = {'source_run': 'bd4c107ad9654a1288761a207dcd7d98', 'scored_documents': len(docs), 'scored_examples': len(rows), 'bootstrap_repetitions': len(boot), 'bootstrap_seed': 20261122, 'observed_delta': 26/1279, 'bootstrap_95_percentile_interval': np.quantile(boot, [0.025, 0.975]).tolist(), 'shuffle_repetitions': len(shuffled), 'shuffle_mean_accuracy': float(np.mean(shuffled)), 'shuffle_95_percentile_interval': np.quantile(shuffled, [0.025, 0.975]).tolist(), 'actual_cue_accuracy': 876/1279, 'singleton_stratum_rows': sum(len(v) for v in groups.values() if len(v) == 1), 'cue_count_strata': strata}
    output_rows = [{k: v for k, v in r.items() if k not in ('cues', 'predict')} for r in rows]
    mlflow.set_experiment('word-energy-natural-context-evaluation')
    with mlflow.start_run(run_name='semcor-paired-document-controls') as run:
        log_contract(mlflow, RunContract('classical', 'cpu_gpu_simulator', 'SemCor paired development controls', 'not applicable', 'word IDs and sense keys', 'not applicable', 'cluster bootstrap and within-target permutation', 20261122, ARCHIVE, '225 fit and 57 validation; outer 70 unopened', 'no graph', 'post-hoc conditional development analysis'), result)
        mlflow.log_dict(result, 'results.json'); mlflow.log_dict(output_rows, 'paired_rows.json')
        for p in (Path(__file__), ROOT/'notes/SEMCOR_PAIRED_CONTROL_PROTOCOL.md'):
            mlflow.log_artifact(str(p), 'source_snapshot')
        print('RUN', run.info.run_id, json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
