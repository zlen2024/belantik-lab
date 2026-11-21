"""Inspect SemCor anchor-fit documents only, leaving reserved documents closed."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json,re,sys,zipfile,xml.etree.ElementTree as ET
import mlflow
from mlflow.tracking import MlflowClient
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file

EXPECTED='126fa2e829ab63edd5b3fd9de45ef1d60d6880e01e25abc55b5ac7918a824655';VOCAB_RUN='3387fc4fb99646f1b7aacd518f39bf10'
def main():
    path=ROOT/'data/semcor/semcor.zip'
    if sha256_file(path)!=EXPECTED:raise ValueError('archive mismatch')
    mlflow.set_tracking_uri(tracking_uri());client=MlflowClient();vr=artifact_dir(client,VOCAB_RUN)
    meta=json.loads((vr/'config.json').read_text(encoding='utf8'));v=meta['vocabulary']
    digest=hashlib.sha256(json.dumps(v,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    if digest!=meta['vocabulary_sha256']:raise ValueError('vocabulary hash')
    vocab={w:i for i,w in enumerate(v)};stop=set(v[:30]);counts=Counter();senses=defaultdict(lambda:defaultdict(list))
    with zipfile.ZipFile(path) as z:
        names=sorted(x for x in z.namelist() if '/tagfiles/' in x and x.endswith('.xml'))
        if len(names)!=352 or len({Path(x).name for x in names})!=352:raise ValueError('document identity')
        key=lambda x:hashlib.sha256(('semcor-fit-v1:'+Path(x).name).encode()).hexdigest()
        ordered=sorted(names,key=key);fit=ordered[:282];reserved=ordered[282:]
        for name in fit:
            try:root=ET.fromstring(z.read(name))
            except ET.ParseError as e:raise ValueError('invalid XML '+name) from e
            doc=Path(name).stem
            for sent_index,sentence in enumerate(root.iter('s')):
                elements=list(sentence)
                for i,node in enumerate(elements):
                    if node.tag!='wf' or node.get('cmd')!='done' or not (node.get('pos') or '').startswith('NN'):continue
                    counts['annotated_noun_occurrences']+=1
                    surface=(node.text or '').lower();lemma=(node.get('lemma') or '').lower();sense=node.get('lexsn') or '';wnsn=node.get('wnsn') or ''
                    if not re.fullmatch('[a-z]+',surface) or surface!=lemma or surface not in vocab:counts['filtered_surface_or_vocab']+=1;continue
                    if not re.fullmatch(r'[0-9]:[0-9]{2}:[0-9]{2}:[a-z_]*:[0-9]*',sense) or not re.fullmatch(r'[1-9][0-9]*',wnsn):counts['filtered_ambiguous_sense']+=1;continue
                    nearby=[]
                    for delta in sorted((x for x in range(-8,9) if x),key=lambda x:(abs(x),x>0)):
                        j=i+delta
                        if not 0<=j<len(elements):continue
                        c=elements[j]
                        if c.tag!='wf':continue
                        w=(c.text or '').lower()
                        if re.fullmatch('[a-z]+',w) and w in vocab and w not in stop:nearby.append(vocab[w])
                    if len(nearby)<2:counts['filtered_insufficient_context']+=1;continue
                    counts['eligible_occurrences']+=1
                    senses[surface][sense].append({'document':doc,'sentence_index':sent_index,'token_index':i,'context':nearby[:4]})
    anchors=[]
    for word,by_sense in sorted(senses.items()):
        eligible=[(sense,items) for sense,items in by_sense.items() if len(items)>=5 and len({r['document'] for r in items})>=3]
        if len(eligible)<2:continue
        selected=sorted(eligible,key=lambda x:(-len(x[1]),x[0]))[:2]
        anchors.append({'word':word,'target_id':vocab[word],'senses':[{'sense_key':word+'%'+sense,'occurrences':len(items),'documents':len({r['document'] for r in items}),'items':items} for sense,items in selected]})
    summary={'archive_sha256':EXPECTED,'vocabulary_run':VOCAB_RUN,'vocabulary_sha256':digest,'documents_total':352,'fit_documents':len(fit),'reserved_documents':len(reserved),'counts':dict(counts),'candidate_targets':len(anchors),
      'targets':[{'word':a['word'],'senses':[{'sense_key':s['sense_key'],'occurrences':s['occurrences'],'documents':s['documents']} for s in a['senses']]} for a in anchors],
      'fit_document_hashes':[key(x) for x in fit],'reserved_document_hashes':[key(x) for x in reserved],
      'limitations':'training labels only; WordNet mapping automatic; lexical POS/surface/vocab filtering; context sufficiency and corpus overlap unknown; reserved XML not read; no model score'}
    mlflow.set_experiment('word-energy-natural-context-evaluation')
    with mlflow.start_run(run_name='semcor-train-anchor-feasibility') as run:
        log_contract(mlflow,RunContract('classical','cpu_gpu_simulator','semcor train-only anchor audit','no model','word IDs and WordNet sense keys','not applicable','document hash split and deterministic counts',0,EXPECTED,'SemCor anchor-fit documents only; 70 reserved unread','no graph',summary['limitations']),summary)
        mlflow.log_dict(summary,'summary.json');mlflow.log_dict(anchors,'anchors.json')
        for p in (Path(__file__),ROOT/'notes/SEMCOR_TRAIN_ANCHOR_FEASIBILITY_PROTOCOL.md'):mlflow.log_artifact(str(p),'source_snapshot')
        print('RUN',run.info.run_id,json.dumps({k:v for k,v in summary.items() if k not in ('targets','fit_document_hashes','reserved_document_hashes')}),flush=True)
if __name__=='__main__':main()
