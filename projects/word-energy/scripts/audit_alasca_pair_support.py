"""Audit explicit silver preference-pair support; read ALaSca training entry only."""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
import mlflow

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_wikitext2_joint_context_data import load
from scripts.evaluate_swords_external_candidates import DATA
from word_energy.tracking import RunContract,log_contract,sha256_file,tracking_uri

EXPECTED='0198abbe7f39ac6b5aa67ed6caf389795ad684cd9a2e0339ecc9e3640c55545b'
def main():
    source = sys.argv[1] if len(sys.argv) == 2 else 'alasca'
    if source not in ('alasca', 'twsi') or len(sys.argv) > 2: raise ValueError('source must be alasca or twsi')
    path=ROOT/'data/alasca/alasca_dataset.zip'
    if sha256_file(path)!=EXPECTED: raise ValueError('archive hash mismatch')
    _, meta=load(DATA)
    vocab={w:i for i,w in enumerate(meta['vocabulary'])}
    counts=Counter(); cells=defaultdict(lambda: defaultdict(set)); sentences=set()
    with zipfile.ZipFile(path) as z, z.open(f'alasca_datasets/{source}_train.tsv') as raw:
        reader=csv.reader(io.TextIOWrapper(raw,encoding='utf8'),delimiter='\t')
        for row in reader:
            counts['rows']+=1
            if len(row)!=6: counts['invalid_columns']+=1; continue
            lexeme,instance,index,sentence,mask,gold=row
            tokens=sentence.split()
            try: i=int(index)
            except ValueError: counts['invalid_index']+=1; continue
            if not 0<=i<len(tokens): counts['invalid_index']+=1; continue
            target=tokens[i].lower()
            if not re.fullmatch('[a-z]+',target) or target not in vocab: counts['target_oov']+=1; continue
            ctx=next((tokens[j].lower() for j in (i-1,i+1) if 0<=j<len(tokens) and re.fullmatch('[a-z]+',tokens[j].lower()) and tokens[j].lower() in vocab),None)
            if ctx is None: counts['no_context']+=1; continue
            scored={}; malformed=False
            for item in gold.split():
                try: word,s=item.rsplit('::',1); v=float(s)
                except ValueError: malformed=True; break
                import math
                word=word.lower()
                if not math.isfinite(v): malformed=True; break
                if re.fullmatch('[a-z]+',word) and word in vocab and word != target:
                    if word in scored: malformed=True; break
                    scored[word]=v
            if malformed: counts['invalid_gold']+=1; continue
            counts['usable_rows']+=1
            senthash=hashlib.sha256(' '.join(t.casefold() for t in tokens).encode()).hexdigest()
            sentences.add(senthash)
            for a,b in combinations(sorted(scored),2):
                if scored[a]==scored[b]: continue
                counts['strict_pair_rows']+=1
                cells[(vocab[target],vocab[a],vocab[b])][(vocab[ctx], scored[a]>scored[b])].add(senthash)
    reversals=0; eligible_groups=0; contextual_cells=0
    for groups in cells.values():
        contextual_cells+=len(groups)
        pos=[(ctx,h) for (ctx,sign),h in groups.items() if sign]
        neg=[(ctx,h) for (ctx,sign),h in groups.items() if not sign]
        n=sum(1 for c1,h1 in pos for c2,h2 in neg if c1!=c2 and any(x!=y for x in h1 for y in h2))
        reversals+=n; eligible_groups+=bool(n)
    result={'counts':dict(counts),'unique_sentence_hashes_usable':len(sentences),
      'target_candidate_pair_groups':len(cells),'context_preference_cells':contextual_cells,
      'reversal_cell_pairs':reversals,'groups_with_reversal':eligible_groups,
      'source':source,'limitations':'one token context may not justify label; no document IDs, no model evaluation or training; official test and dev entries unopened; alasca scores are automatic silver; twsi scores originate from human substitute lists'}
    mlflow.set_tracking_uri(tracking_uri()); mlflow.set_experiment('word-energy-natural-context-evaluation')
    with mlflow.start_run(run_name=f'{source}-training-pair-support-audit') as run:
        log_contract(mlflow,RunContract('classical','cpu_gpu_simulator','dataset support audit','no model','token IDs','not applicable','deterministic scan',0,EXPECTED,source+' training only','no graph',result['limitations']),result)
        mlflow.log_dict(result,'summary.json')
        protocol = 'TWSI_TRAIN_PAIR_SUPPORT_PROTOCOL.md' if source == 'twsi' else 'ALASCA_TRAIN_PAIR_SUPPORT_PROTOCOL.md'
        for p in (Path(__file__),ROOT/'notes'/protocol):mlflow.log_artifact(str(p),'source_snapshot')
        print('RUN',run.info.run_id,json.dumps(result),flush=True)

if __name__=='__main__':main()
