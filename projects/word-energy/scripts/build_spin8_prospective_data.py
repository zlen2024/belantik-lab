"""Prospectively reserve WikiText train articles and build fit-only vocabulary."""
from collections import Counter
from pathlib import Path
import hashlib,json,sys,tempfile
import mlflow
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_wikitext2_document_data import parse,corpus_data,sample,norm,FILES
from word_energy.text8_development import sha_array,noise_rows
from word_energy.tracking import RunContract,log_contract,tracking_uri

def main():
    docs=parse('train')
    key=lambda d:hashlib.sha256(('spin8-holdout-v1:'+d['title']).encode()).hexdigest()
    reserved={key(d) for d in sorted(docs,key=key)[:60]}
    fit=[d for d in docs if key(d) not in reserved];held=[d for d in docs if key(d) in reserved]
    if len(fit)!=540 or len(held)!=60:raise ValueError('bad article split')
    whole=lambda ds:{hashlib.sha256(norm(' '.join(d['rows'])).encode()).hexdigest() for d in ds}
    paragraphs=lambda ds:{norm(r) for d in ds for r in d['rows'] if len(norm(r))>=80}
    if whole(fit)&whole(held) or paragraphs(fit)&paragraphs(held):raise ValueError('exact document or paragraph overlap')
    raw,_=corpus_data(fit);count=Counter(x for x in raw if isinstance(x,str))
    vocabulary=sorted((w for w,c in count.items() if c>=5),key=lambda w:(-count[w],w))[:2000]
    if len(vocabulary)!=2000:raise ValueError('insufficient fit vocab')
    vocab={w:i for i,w in enumerate(vocabulary)};counts=np.array([count[w] for w in vocabulary],dtype=np.int64)
    noise_q=counts.astype(float)**.75;noise_q/=noise_q.sum()
    fit_raw,fit_docs=corpus_data(fit,vocab);held_raw,held_docs=corpus_data(held,vocab)
    fit_ids=np.asarray(fit_raw,dtype=np.int32);held_ids=np.asarray(held_raw,dtype=np.int32)
    tp,tpos,tdoc=sample(fit_ids,fit_docs,10000,np.random.default_rng(89101))
    vp,vpos,vdoc=sample(held_ids,held_docs,2000,np.random.default_rng(89102))
    tn=np.stack([noise_rows(noise_q,tp[:,0],5,np.random.default_rng(89110+e)) for e in range(3)])
    vn=noise_rows(noise_q,vp[:,0],5,np.random.default_rng(89103))
    orders=np.stack([np.random.default_rng(89120+e).permutation(len(tp)).astype(np.int32) for e in range(3)])
    data={'train_pairs':tp,'train_positions':tpos,'train_document':tdoc,'train_noise':tn,'orders':orders,
          'holdout_pairs':vp,'holdout_positions':vpos,'holdout_document':vdoc,'holdout_noise':vn,'counts':counts,'noise_q':noise_q}
    def coverage(ids,doc):return float(np.sum((ids>=0)&(doc>=0))/np.sum(doc>=0))
    meta={'source_sha256':FILES['train'],'vocabulary':vocabulary,'vocabulary_sha256':hashlib.sha256(json.dumps(vocabulary,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
      'fit_title_hashes':[key(d) for d in fit],'holdout_title_hashes':[key(d) for d in held],
      'fit_articles':len(fit),'holdout_articles':len(held),'fit_token_coverage':coverage(fit_ids,fit_docs),'holdout_token_coverage':coverage(held_ids,held_docs),
      'train_unique_position_pairs':int(len(np.unique(tpos,axis=0))),'holdout_unique_position_pairs':int(len(np.unique(vpos,axis=0))),
      'fit_documents_sampled':int(len(np.unique(tdoc))),'holdout_documents_sampled':int(len(np.unique(vdoc))),
      'train_exact_noise_collisions':[int(np.sum(tn[e]==tp[:,0,None])) for e in range(3)],'holdout_exact_noise_collisions':int(np.sum(vn==vp[:,0,None])),
      'hashes':{k:sha_array(v) for k,v in data.items()},'limits':'prospective WikiText train article holdout; historical corpus exposure, near duplicates and context semantics not excluded; no official validation/test'}
    mlflow.set_tracking_uri(tracking_uri());mlflow.set_experiment('word-energy-spin8-prospective')
    with mlflow.start_run(run_name='spin8-prospective-data') as run:
        log_contract(mlflow,RunContract('classical','cpu_gpu_simulator','prospective article split','no model','token IDs','not applicable','seeded lexical pair sampling',89101,FILES['train'],'fit 540 / heldout 60 source train articles','no graph',meta['limits']),meta)
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as d:
            p=Path(d)/'dataset.npz';np.savez_compressed(p,**data);mlflow.log_artifact(str(p),'dataset')
        for p in (Path(__file__),ROOT/'scripts/build_wikitext2_document_data.py',ROOT/'notes/SPIN_6_VS_8_PROSPECTIVE_PROTOCOL.md'):mlflow.log_artifact(str(p),'source_snapshot')
        print('RUN',run.info.run_id,json.dumps({k:v for k,v in meta.items() if k not in ('vocabulary','fit_title_hashes','holdout_title_hashes','hashes')}),flush=True)
if __name__=='__main__':main()
