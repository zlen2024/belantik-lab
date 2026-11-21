"""Exact normalized TWSI train sentence subsequences in WikiText training rows."""
from collections import defaultdict
from pathlib import Path
import csv, io, json, re, sys, zipfile
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from word_energy.tracking import RunContract,log_contract,sha256_file,tracking_uri

def tokens(s): return tuple(re.findall(r'[a-z0-9]+',s.lower()))

def find_matches(index, text):
    """Find full token sequences within one normalized text row."""
    row=tokens(text or '')
    return {s for i in range(max(0,len(row)-4)) for s in index.get(row[i:i+5],())
            if row[i:i+len(s)]==s}

def main():
    import mlflow
    archive=ROOT/'data/alasca/alasca_dataset.zip'; wiki=ROOT/'data/wikitext2/train.parquet'
    sha=sha256_file(archive)
    if sha!='0198abbe7f39ac6b5aa67ed6caf389795ad684cd9a2e0339ecc9e3640c55545b':raise ValueError('archive hash mismatch')
    sentences=[]; short=0
    with zipfile.ZipFile(archive) as z,z.open('alasca_datasets/twsi_train.tsv') as f:
        for row in csv.reader(io.TextIOWrapper(f,encoding='utf8'),delimiter='\t',quoting=csv.QUOTE_NONE):
            if len(row)!=6:raise ValueError('bad row')
            t=tokens(row[3]); sentences.append(t); short+=len(t)<5
    index=defaultdict(set)
    for s in set(sentences):
        if len(s)>=5:index[s[:5]].add(s)
    hits=set(); rows=set(); nrows=0
    for batch in pq.ParquetFile(wiki).iter_batches(columns=['text'],batch_size=512):
        for text in batch.column(0).to_pylist():
            found=find_matches(index,text); nrows+=1
            if found: hits.update(found); rows.add(nrows-1)
    result={'twsi_rows':len(sentences),'twsi_unique_sentences':len(set(sentences)),
       'sentences_shorter_than_5_tokens_excluded':short,'twsi_unique_matched':len(hits),
       'twsi_rows_matched':sum(s in hits for s in sentences),'wikitext_rows_matched':len(rows),'wikitext_rows_scanned':nrows,
       'archive_sha256':sha,'wikitext_train_sha256':sha256_file(wiki),
       'limitations':'exact normalized contiguous same-parquet-row only; misses short, near-duplicates and shared documents; no model, labels or held-out data'}
    mlflow.set_tracking_uri(tracking_uri());mlflow.set_experiment('word-energy-natural-context-evaluation')
    with mlflow.start_run(run_name='twsi-wikitext-train-exact-overlap-audit') as run:
        log_contract(mlflow,RunContract('classical','cpu_gpu_simulator','training corpus overlap audit','no model','normalized tokens','not applicable','deterministic subsequence',0,sha,'TWSI train and WikiText train only','no graph',result['limitations']),result)
        mlflow.log_dict(result,'summary.json')
        for p in (Path(__file__),ROOT/'notes/TWSI_WIKITEXT_OVERLAP_PROTOCOL.md'):mlflow.log_artifact(str(p),'source_snapshot')
        print('RUN',run.info.run_id,json.dumps(result),flush=True)
if __name__=='__main__':main()
