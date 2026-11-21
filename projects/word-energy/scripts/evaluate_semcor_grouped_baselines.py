"""Frozen SemCor inner document-split baseline; outer reserved XML remains closed."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri

ARCHIVE = '126fa2e829ab63edd5b3fd9de45ef1d60d6880e01e25abc55b5ac7918a824655'
VOCAB_RUN = '3387fc4fb99646f1b7aacd518f39bf10'

def key(prefix, name):
    return hashlib.sha256((prefix + Path(name).name).encode()).hexdigest()

def collect(z, names, vocab, stop):
    result = defaultdict(lambda: defaultdict(list))
    for name in names:
        root = ET.fromstring(z.read(name))
        for si, sentence in enumerate(root.iter('s')):
            elements = list(sentence)
            for i, node in enumerate(elements):
                if node.tag != 'wf' or node.get('cmd') != 'done' or node.get('pos') != 'NN':
                    continue
                word = (node.text or '').lower()
                sense, wnsn = node.get('lexsn') or '', node.get('wnsn') or ''
                if not re.fullmatch('[a-z]+', word) or word != (node.get('lemma') or '').lower() or word not in vocab:
                    continue
                if not re.fullmatch(r'[0-9]:[0-9]{2}:[0-9]{2}:[a-z_]*:[0-9]*', sense) or not re.fullmatch(r'[1-9][0-9]*', wnsn):
                    continue
                cues = []
                for delta in sorted((d for d in range(-8, 9) if d), key=lambda d: (abs(d), d > 0)):
                    j = i + delta
                    if not 0 <= j < len(elements) or elements[j].tag != 'wf':
                        continue
                    w = (elements[j].text or '').lower()
                    if re.fullmatch('[a-z]+', w) and w in vocab and w not in stop and w != word:
                        cues.append(vocab[w])
                cues = cues[:4]
                if len(set(cues)) < 2:
                    continue
                result[word][sense].append({'doc': Path(name).stem, 'sentence': si, 'token': i, 'cues': cues})
    return result

def main():
    archive = ROOT / 'data/semcor/semcor.zip'
    assert sha256_file(archive) == ARCHIVE
    mlflow.set_tracking_uri(tracking_uri())
    meta = json.loads((artifact_dir(MlflowClient(), VOCAB_RUN) / 'config.json').read_text(encoding='utf8'))
    words = meta['vocabulary']
    assert hashlib.sha256(json.dumps(words, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest() == meta['vocabulary_sha256']
    vocab, stop = {w: i for i, w in enumerate(words)}, set(words[:30])
    with zipfile.ZipFile(archive) as z:
        names = [x for x in z.namelist() if '/tagfiles/' in x and x.endswith('.xml')]
        assert len(names) == 352 and len({Path(x).name for x in names}) == 352
        outer = sorted(names, key=lambda x: key('semcor-fit-v1:', x))[:282]
        inner = sorted(outer, key=lambda x: key('semcor-inner-v1:', x))
        train_names, val_names = inner[:225], inner[225:]
        assert len(val_names) == 57 and not set(train_names) & set(val_names)
        train = collect(z, train_names, vocab, stop)
        val = collect(z, val_names, vocab, stop)
    per_target = []
    selected_count = other_count = candidate_count = 0
    for word, senses in sorted(train.items()):
        qualified = [(s, xs) for s, xs in senses.items() if len(xs) >= 5 and len({x['doc'] for x in xs}) >= 3]
        if len(qualified) < 2:
            continue
        candidate_count += 1
        chosen = sorted(qualified, key=lambda p: (-len(p[1]), p[0]))[:2]
        keys = sorted(s for s, _ in chosen)
        items = dict(chosen)
        counts = {s: len(items[s]) for s in keys}
        majority = sorted(keys, key=lambda s: (-counts[s], s))[0]
        token_counts = {s: Counter(c for x in items[s] for c in x['cues']) for s in keys}
        total = {s: sum(token_counts[s].values()) for s in keys}
        correct = {'majority': Counter(), 'cues': Counter()}
        n = Counter()
        for sense, examples in val.get(word, {}).items():
            if sense not in keys:
                other_count += len(examples)
                continue
            for example in examples:
                n[sense] += 1
                logp = {s: math.log(counts[s] / sum(counts.values())) + sum(math.log((token_counts[s][c] + 1) / (total[s] + len(words))) for c in example['cues']) for s in keys}
                pred = sorted(keys, key=lambda s: (-logp[s], s))[0]
                correct['majority'][sense] += int(majority == sense)
                correct['cues'][sense] += int(pred == sense)
        if not sum(n.values()):
            continue
        selected_count += sum(n.values())
        per_target.append({'word': word, 'senses': keys, 'train_counts': counts, 'validation_counts': dict(n), 'correct': {k: dict(v) for k, v in correct.items()}})
    assert per_target and selected_count
    both = [r for r in per_target if all(r['validation_counts'].get(s, 0) for s in r['senses'])]
    def metrics(method):
        return {'accuracy': sum(sum(r['correct'][method].values()) for r in per_target) / selected_count,
                'target_macro_accuracy': sum(sum(r['correct'][method].values()) / sum(r['validation_counts'].values()) for r in per_target) / len(per_target),
                'target_macro_balanced_accuracy': (sum(sum(r['correct'][method].get(s, 0) / r['validation_counts'][s] for s in r['senses']) / 2 for r in both) / len(both)) if both else None}
    result = {'archive_sha256': ARCHIVE, 'vocabulary_sha256': meta['vocabulary_sha256'], 'inner_fit_documents': len(train_names), 'inner_validation_documents': len(val_names), 'outer_reserved_documents_unopened': 70,
              'train_selected_targets': candidate_count, 'selected_validation_occurrences': selected_count, 'other_sense_validation_occurrences_excluded': other_count, 'validation_targets': len(per_target), 'targets_with_both_senses': len(both),
              'majority': metrics('majority'), 'cues': metrics('cues'), 'per_target': per_target,
              'inner_fit_document_hashes': [key('semcor-inner-v1:', n) for n in train_names], 'inner_validation_document_hashes': [key('semcor-inner-v1:', n) for n in val_names]}
    result['cues_minus_majority'] = {k: (result['cues'][k] - result['majority'][k]) if result['cues'][k] is not None else None for k in result['cues']}
    mlflow.set_experiment('word-energy-natural-context-evaluation')
    with mlflow.start_run(run_name='semcor-grouped-dev-baselines') as run:
        log_contract(mlflow, RunContract('classical', 'cpu_gpu_simulator', 'SemCor inner grouped development baselines', 'majority and multinomial naive Bayes', 'word IDs and sense keys', 'not applicable', 'deterministic document split', 0, ARCHIVE, '225 fit, 57 inner validation; outer 70 unopened', 'no graph', 'development baselines, selected senses only; no EBM evaluation'), result)
        mlflow.log_dict(result, 'results.json')
        for p in (Path(__file__), ROOT / 'notes/SEMCOR_GROUPED_VALIDATION_PROTOCOL.md'):
            mlflow.log_artifact(str(p), 'source_snapshot')
        print('RUN', run.info.run_id, json.dumps({k: v for k, v in result.items() if k not in ('per_target', 'inner_fit_document_hashes', 'inner_validation_document_hashes')}), flush=True)

if __name__ == '__main__':
    main()
