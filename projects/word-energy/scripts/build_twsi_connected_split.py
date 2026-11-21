"""Build an exploratory connected-group split from TWSI weak preferences."""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import csv, hashlib, io, json, math, re, sys, tempfile, zipfile
import mlflow
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_wikitext2_joint_context_data import load
from scripts.evaluate_swords_external_candidates import DATA
from word_energy.tracking import RunContract,log_contract,sha256_file,tracking_uri
from word_energy.text8_development import sha_array

EXPECTED='0198abbe7f39ac6b5aa67ed6caf389795ad684cd9a2e0339ecc9e3640c55545b'

def components(rows):
    parent=list(range(len(rows)))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        x,y=find(a),find(b)
        if x!=y:parent[y]=x
    seen_sentence={};seen_pair={}
    for i,r in enumerate(rows):
        for mapping,key in ((seen_sentence,r[5]),(seen_pair,(r[0],r[2],r[3]))):
            if key in mapping:union(i,mapping[key])
            else:mapping[key]=i
    groups=defaultdict(list)
    for i in range(len(rows)):groups[find(i)].append(i)
    return list(groups.values())

def main():
    path=ROOT/'data/alasca/alasca_dataset.zip'
    if sha256_file(path)!=EXPECTED:raise ValueError('archive hash mismatch')
    _,metadata=load(DATA);vocab={w:i for i,w in enumerate(metadata['vocabulary'])}
    examples=[];counts=Counter()
    with zipfile.ZipFile(path) as z,z.open('alasca_datasets/twsi_train.tsv') as raw:
        for row in csv.reader(io.TextIOWrapper(raw,encoding='utf8'),delimiter='\t',quoting=csv.QUOTE_NONE):
            counts['source_rows']+=1
            if len(row)!=6:raise ValueError('invalid fields')
            try:i=int(row[2])
            except ValueError:continue
            words=row[3].split()
            if not 0<=i<len(words):continue
            target=words[i].lower()
            if not re.fullmatch('[a-z]+',target) or target not in vocab:continue
            cue=next((words[j].lower() for j in (i-1,i+1) if 0<=j<len(words) and re.fullmatch('[a-z]+',words[j].lower()) and words[j].lower() in vocab),None)
            if cue is None:continue
            scores={};invalid=False
            for item in row[5].split():
                try:w,v=item.rsplit('::',1);v=float(v)
                except ValueError:invalid=True;break
                w=w.lower()
                if not math.isfinite(v):invalid=True;break
                if re.fullmatch('[a-z]+',w) and w in vocab and w!=target:
                    if w in scores:invalid=True;break
                    scores[w]=v
            if invalid:continue
            senthash=hashlib.sha256(' '.join(w.casefold() for w in words).encode()).hexdigest()
            for a,b in combinations(sorted(scores),2):
                if scores[a]!=scores[b]:examples.append((vocab[target],vocab[cue],vocab[a],vocab[b],int(scores[a]>scores[b]),senthash))
    counts['strict_examples_before_dedup']=len(examples)
    examples=sorted(set(examples));counts['strict_examples_after_dedup']=len(examples)
    directions=defaultdict(set)
    for r in examples:directions[r[:4]].add(r[4])
    conflicts={k for k,v in directions.items() if len(v)>1}
    rows=[r for r in examples if r[:4] not in conflicts]
    counts['conflicting_visible_inputs']=len(conflicts);counts['examples_after_conflict_filter']=len(rows)
    groups=components(rows);partition={}
    for group in groups:
        ids=sorted({rows[i][5] for i in group})
        keys=sorted({(rows[i][0],rows[i][2],rows[i][3]) for i in group})
        raw=json.dumps({'sentences':ids,'pairs':keys},separators=(',',':'),sort_keys=True).encode()
        digest=hashlib.sha256(raw).hexdigest();split=int(digest[:16],16)%5==0
        for i in group:partition[i]=int(split)
    train=np.array([i for i in range(len(rows)) if partition[i]==0],dtype=np.int64)
    val=np.array([i for i in range(len(rows)) if partition[i]==1],dtype=np.int64)
    for keys in (lambda r:r[5],lambda r:(r[0],r[2],r[3])):
        if {keys(rows[i]) for i in train}&{keys(rows[i]) for i in val}:raise ValueError('cross-split group')
    def describe(ix):
        pairs=defaultdict(set)
        for i in ix:
            r=rows[i];pairs[(r[0],r[2],r[3])].add(r[4])
        return {'examples':len(ix),'targets':len({rows[i][0] for i in ix}),'pair_groups':len(pairs),
                'bidirectional_groups':sum(len(v)==2 for v in pairs.values()),'sentences':len({rows[i][5] for i in ix})}
    arr=np.array([r[:5] for r in rows],dtype=np.int32)
    sentence_hashes=np.array([r[5] for r in rows],dtype='U64')
    summary={'counts':dict(counts),'components':len(groups),'largest_component_examples':max(map(len,groups),default=0),
      'train':describe(train),'validation':describe(val),'overlapping_target_types':len({rows[i][0] for i in train}&{rows[i][0] for i in val}),
      'hashes':{'examples':sha_array(arr),'sentence_hashes':sha_array(sentence_hashes),'train_indices':sha_array(train),'validation_indices':sha_array(val)},
      'data_run':DATA,'limits':'posthoc train-only weak preferences; exact sentence and target-candidate pairs grouped, not document-disjoint; cue sufficiency, source and near-duplicates unverified; no model or official test'}
    mlflow.set_tracking_uri(tracking_uri());mlflow.set_experiment('word-energy-natural-context-evaluation')
    with mlflow.start_run(run_name='twsi-connected-group-split-feasibility') as run:
        log_contract(mlflow,RunContract('classical','cpu_gpu_simulator','twsi exploratory split','no model','word IDs','not applicable','deterministic connected components',0,EXPECTED,'TWSI released train only','no graph',summary['limits']),summary)
        mlflow.log_dict(summary,'summary.json');tmp=ROOT/'.tmp';tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as d:
            p=Path(d)/'split.npz';np.savez_compressed(p,examples=arr,sentence_hashes=sentence_hashes,train_indices=train,validation_indices=val);mlflow.log_artifact(str(p),'dataset')
        for p in (Path(__file__),ROOT/'notes/TWSI_CONNECTED_SPLIT_PROTOCOL.md'):mlflow.log_artifact(str(p),'source_snapshot')
        print('RUN',run.info.run_id,json.dumps(summary),flush=True)
if __name__=='__main__':main()
