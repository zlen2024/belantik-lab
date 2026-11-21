"""Construct a label-blinded, train-only AI cue review panel."""
from collections import defaultdict
from pathlib import Path
import csv,hashlib,io,json,sys,zipfile
import numpy as np
from mlflow.tracking import MlflowClient
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.build_wikitext2_joint_context_data import load
from scripts.evaluate_swords_external_candidates import DATA
from word_energy.text8_development import sha_array
from word_energy.tracking import tracking_uri,sha256_file

def main():
    client=MlflowClient(tracking_uri=tracking_uri())
    root=artifact_dir(client,'16e0bf36c0574dcb92cd88cdf1a9b1d3')
    config=json.loads((root/'config.json').read_text(encoding='utf8'))
    with np.load(root/'dataset/split.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
    if {k:sha_array(v) for k,v in a.items()}!=config['hashes']:raise ValueError('split mismatch')
    cr=artifact_dir(client,'8f2b8e9cfa8344d6bed4c2155e003cda')
    cconfig=json.loads((cr/'summary.json').read_text(encoding='utf8'))
    with np.load(cr/'dataset/contexts.npz',allow_pickle=False) as f:ctx=f['contexts']
    if sha_array(ctx)!=cconfig['contexts_sha256']:raise ValueError('context mismatch')
    _,meta=load(DATA);v=meta['vocabulary'];n=len(v)
    groups=defaultdict(lambda:defaultdict(list));rows=a['examples'];sh=a['sentence_hashes']
    for i in a['train_indices']:
        r=rows[i];groups[(int(r[0]),int(r[2]),int(r[3]))][int(r[4])].append(int(i))
    candidates=[]
    for key,sides in sorted(groups.items()):
        if 0 not in sides or 1 not in sides:continue
        eligible=[(i,j) for i in sides[0] for j in sides[1] if sh[i]!=sh[j] and sorted(ctx[i].tolist())!=sorted(ctx[j].tolist())]
        if eligible:candidates.append((key,sorted(eligible)))
    order=np.random.default_rng(88121).permutation(len(candidates));chosen=[];seen=set()
    for j in order:
        key,pairs=candidates[j]
        if key[0] in seen:continue
        chosen.append((key,pairs[0]));seen.add(key[0])
        if len(chosen)==12:break
    if len(chosen)!=12:raise ValueError('insufficient distinct targets')
    needed={str(sh[i]) for _,pair in chosen for i in pair};lookup=defaultdict(set)
    archive=ROOT/'data/alasca/alasca_dataset.zip'
    if sha256_file(archive)!='0198abbe7f39ac6b5aa67ed6caf389795ad684cd9a2e0339ecc9e3640c55545b':raise ValueError('archive mismatch')
    with zipfile.ZipFile(archive) as z,z.open('alasca_datasets/twsi_train.tsv') as raw:
        for row in csv.reader(io.TextIOWrapper(raw,encoding='utf8'),delimiter='\t',quoting=csv.QUOTE_NONE):
            if len(row)!=6:raise ValueError('bad row')
            sentence=row[3];h=hashlib.sha256(' '.join(w.casefold() for w in sentence.split()).encode()).hexdigest()
            if h in needed:lookup[h].add(sentence)
    if any(len(lookup[h])!=1 for h in needed):raise ValueError('sentence recovery ambiguous')
    cue=[];full=[];keydata=[];swap=np.random.default_rng(88122)
    for g,(key,pair) in enumerate(chosen):
        target,ca,cb=key;flipped=bool(swap.integers(2));options=[v[ca],v[cb]]
        if flipped:options.reverse()
        for side,i in enumerate(pair):
            identifier='item-'+hashlib.sha256(f'88124:{i}'.encode()).hexdigest()[:12]
            common={'id':identifier,'target':v[target],'A':options[0],'B':options[1]}
            cue.append({**common,'context_bag':sorted(v[int(x)] for x in ctx[i] if x<n)})
            full.append({**common,'sentence':next(iter(lookup[str(sh[i])]))})
            prefer_a=bool(rows[i,4]);prefer_a=not prefer_a if flipped else prefer_a
            keydata.append({'id':identifier,'group':g,'row_index':i,'weak_preference':'A' if prefer_a else 'B','sentence_hash':str(sh[i])})
    order=np.random.default_rng(88123).permutation(len(cue))
    out=ROOT/'.tmp/twsi_blind_review';out.mkdir(parents=True,exist_ok=True)
    for name,obj in [('cue.json',[cue[i] for i in order]),('full.json',[full[i] for i in order]),('key.json',[keydata[i] for i in order])]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'panel_rows':len(cue),'groups':len(chosen),'targets':len(seen),'output':str(out),'hashes':{name:sha256_file(out/name) for name in ('cue.json','full.json','key.json')}}))
if __name__=='__main__':main()
