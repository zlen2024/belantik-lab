"""Document-separated WikiText-2 lexical sample preparation."""
from pathlib import Path
from collections import Counter
import hashlib,json,re,sys,tempfile
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.audit_text8_canonical_evaluator import artifact_dir
from word_energy.text8_development import sha_array,noise_rows
from word_energy.text8_lexical_models import pair_metrics
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file

FILES={'train':'49280f3dd993cbdfd322186194133d5d1935f81e1c91064c239f2a70ebb02701','validation':'a10eb5a42c4264c45dd38bfef4da6a74ddef5fc22cb419e1a89d54c4cc6d748f'}
HEAD=re.compile(r'^\s*=\s*([^=\n]+?)\s*=\s*$');SECTION=re.compile(r'^\s*(?:=\s*){2,}[^=\n]+(?:\s*=\s*){2,}$');TOK=re.compile(r'[a-z]+')
URI='https://huggingface.co/datasets/Salesforce/wikitext/tree/c937478d5e4c5aca1b3f9f12de797092d0fdfdef/wikitext-2-raw-v1'

def parse(split):
    p=ROOT/'data/wikitext2'/f'{split}.parquet'
    if sha256_file(p)!=FILES[split]:raise ValueError('archive source SHA mismatch')
    rows=pq.read_table(p,columns=['text']).column('text').to_pylist();docs=[];current=None
    for i,row in enumerate(rows):
        match=HEAD.match(row)
        is_title=bool(match and i>0 and i+1<len(rows) and not rows[i-1].strip() and not rows[i+1].strip())
        if is_title:
            if current is not None:docs.append(current)
            current={'title':match.group(1).strip().casefold(),'rows':[]}
        elif row.strip() and not SECTION.match(row):
            if current is None:raise ValueError('text before first article heading')
            current['rows'].append(row)
    if current is not None:docs.append(current)
    expected=600 if split=='train' else 60
    if len(docs)!=expected or len({d['title'] for d in docs})!=expected:raise ValueError(f'article boundary mismatch {split}: {len(docs)}')
    return docs

def norm(s):return ' '.join(s.casefold().split())
def corpus_data(docs,vocab=None):
    data=[];doc=[]
    for i,d in enumerate(docs):
        for row in d['rows']:
            words=TOK.findall(row.lower())
            if not words:continue
            data.extend(words if vocab is None else [vocab.get(w,-1) for w in words]);data.extend([-1,-1])
            doc.extend([i]*len(words));doc.extend([-1,-1])
    return data,np.asarray(doc,dtype=np.int32)

def sample(ids,doc,n,rng):
    ids=np.asarray(ids,dtype=np.int32);idx=np.arange(len(ids)); offsets=np.array([-2,-1,1,2],dtype=np.int32)
    choices=idx[:,None]+offsets; valid=(choices>=0)&(choices<len(ids));clipped=np.clip(choices,0,len(ids)-1)
    valid &= (ids[:,None]>=0)&(ids[clipped]>=0)&(doc[:,None]==doc[clipped])
    intermediate=np.clip(idx[:,None]+np.sign(offsets)[None,:],0,len(ids)-1)
    valid &= ids[intermediate]>=0
    eligible=np.flatnonzero(valid.any(axis=1))
    if len(eligible)==0:raise ValueError('no eligible positions')
    chosen=rng.choice(eligible,size=n,replace=True); mask=valid[chosen];pick=(rng.random(n)*mask.sum(axis=1)).astype(np.int32)
    offset_idx=np.argmax(np.cumsum(mask,axis=1)>pick[:,None],axis=1); context=chosen+offsets[offset_idx]
    return np.column_stack((ids[chosen],ids[context])).astype(np.int32),np.column_stack((chosen,context)).astype(np.int32),doc[chosen]

def doc_macro(pos,neg,ids):
    wins=((pos[:,None]>neg)+.5*(pos[:,None]==neg)).mean(axis=1)
    groups=np.unique(ids)
    if not len(groups):raise ValueError('empty groups')
    return float(np.mean([wins[ids==i].mean() for i in groups]))

def build():
    train=parse('train');val=parse('validation')
    titles={d['title'] for d in train}&{d['title'] for d in val}
    hashes=lambda ds:{hashlib.sha256(norm(' '.join(d['rows'])).encode()).hexdigest() for d in ds}
    if titles or hashes(train)&hashes(val):raise ValueError('article overlap')
    longrows=lambda ds:{norm(r) for d in ds for r in d['rows'] if len(norm(r))>=80}
    overlap=len(longrows(train)&longrows(val))
    if overlap:raise ValueError(f'exact repeated long paragraph: {overlap}')
    raw,_=corpus_data(train);counts=Counter(x for x in raw if isinstance(x,str));vocab=sorted((w for w,c in counts.items() if c>=5),key=lambda w:(-counts[w],w))[:2000]
    if len(vocab)!=2000:raise ValueError('insufficient vocabulary')
    mapping={w:i for i,w in enumerate(vocab)};count=np.array([counts[w] for w in vocab],dtype=np.int64);q=count.astype(float)**.75;q/=q.sum()
    train_ids,train_doc=corpus_data(train,mapping);val_ids,val_doc=corpus_data(val,mapping)
    train_ids=np.asarray(train_ids,dtype=np.int32);val_ids=np.asarray(val_ids,dtype=np.int32)
    tp,tpos,tdoc=sample(train_ids,train_doc,200000,np.random.default_rng(22101));vp,vpos,vdoc=sample(val_ids,val_doc,10000,np.random.default_rng(22102))
    noises=np.stack([noise_rows(q,tp[:,1],5,np.random.default_rng(22110+e)) for e in range(3)])
    vn=noise_rows(q,vp[:,1],20,np.random.default_rng(22103));orders=np.stack([np.random.default_rng(22120+e).permutation(len(tp)).astype(np.int32) for e in range(3)])
    order=np.random.default_rng(22130).permutation(len(tp)).astype(np.int32);shuffled=tp.copy();shuffled[:,1]=tp[order,1]
    if not np.array_equal(np.sort(order),np.arange(len(tp))) or not np.array_equal(np.sort(tp[:,1]),np.sort(shuffled[:,1])):raise ValueError('invalid shuffle')
    arrays={'train_pairs':tp,'train_positions':tpos,'train_document':tdoc,'shuffled_pairs':shuffled,'shuffle_order':order,
      'validation_pairs':vp,'validation_positions':vpos,'validation_document':vdoc,'train_noise':noises,'validation_noise':vn,'orders':orders,'counts':count,'noise_q':q}
    prior=.25*np.log(count.astype(float));baseline=pair_metrics(prior[vp[:,1]],prior[vn],count[vp[:,0]])
    baseline['document_macro_auc']=doc_macro(prior[vp[:,1]],prior[vn],vdoc)
    stats=lambda x,d:{'raw_tokens':int((d>=0).sum()),'in_vocab_tokens':int((x>=0).sum()),'oov_tokens':int(((x<0)&(d>=0)).sum()),'token_coverage':float((x>=0).sum()/(d>=0).sum())}
    m={'source_uri':URI,'source_sha256':FILES,'license':'CC-BY-SA-3.0 and GFDL as documented in dataset card','articles':{'train':len(train),'validation':len(val),'overlap_titles':len(titles),'overlap_long_paragraphs':overlap},
       'train_title_hashes':sorted(hashlib.sha256(d['title'].encode()).hexdigest() for d in train),'validation_title_hashes':sorted(hashlib.sha256(d['title'].encode()).hexdigest() for d in val),
       'vocabulary':vocab,'train_stats':stats(train_ids,train_doc),'validation_stats':stats(val_ids,val_doc),'validation_documents_sampled':int(len(np.unique(vdoc))),
       'validation_exact_noise_collisions':int(np.sum(vn==vp[:,1,None])), 'train_positive_noise_collisions':[int(np.sum(tp[:,1,None]==x)) for x in noises],
       'shuffle_positive_noise_collisions':[int(np.sum(shuffled[:,1,None]==x)) for x in noises],'shuffle_fixed_indices':int(np.sum(order==np.arange(len(order)))),
       'shuffle_equal_context_ids':int(np.sum(shuffled[:,1]==tp[:,1])),'frequency_only':baseline,'hashes':{k:sha_array(v) for k,v in arrays.items()},
       'limits':'posthoc development; Wikipedia domain shared with text8 and historical overlap unknown; test splits untouched; unequal model capacity'}
    return arrays,m

def load(run_id):
    c=MlflowClient();r=c.get_run(run_id)
    if r.info.status!='FINISHED' or r.data.tags.get('model_family')!='wikitext2_document_data':raise ValueError('invalid data run')
    root=artifact_dir(c,run_id);m=json.loads((root/'config.json').read_text(encoding='utf-8'))
    with np.load(root/'dataset'/'samples.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    if {k:sha_array(v) for k,v in a.items()}!=m['hashes']:raise ValueError('dataset array hash mismatch')
    return a,m

if __name__=='__main__':
    mlflow.set_tracking_uri(tracking_uri());mlflow.set_experiment('word-energy-wikitext2-doc-development');a,m=build()
    source_hash=hashlib.sha256((''.join(FILES.values())).encode()).hexdigest()
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='wikitext2_document_data',energy_convention='no model; fixed unigram prior',variable_encoding='word IDs',beta_temperature_convention='not applicable',sampler_schedule='fixed seeded sample and context permutation',seed=22101,data_hash=source_hash,data_split='WikiText2 raw official train and validation; WikiText2 and text8 test untouched',graph_summary='not applicable',limitations=m['limits'])
    with mlflow.start_run(run_name='wikitext2-document-data') as run:
        log_contract(mlflow,contract,m);tmp=ROOT/'.tmp';tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as d:
            p=Path(d)/'samples.npz';np.savez_compressed(p,**a);mlflow.log_artifact(str(p),'dataset')
        for p in (Path(__file__),ROOT/'notes/WIKITEXT2_DOCUMENT_DEVELOPMENT_PROTOCOL.md'):mlflow.log_artifact(str(p),'source_snapshot')
        print(json.dumps({'run_id':run.info.run_id,'articles':m['articles'],'train_stats':m['train_stats'],'validation_stats':m['validation_stats'],'validation_documents_sampled':m['validation_documents_sampled'],'frequency_only':m['frequency_only']}))
