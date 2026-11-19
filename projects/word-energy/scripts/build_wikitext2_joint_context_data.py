"""Frozen WikiText three-word window construction from existing sampled pairs."""
from pathlib import Path
import hashlib,json,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.build_wikitext2_document_data import load as base_load,parse,corpus_data
from scripts.audit_text8_canonical_evaluator import artifact_dir
from word_energy.text8_development import sha_array,noise_rows
from word_energy.text8_lexical_models import pair_metrics
from scripts.build_wikitext2_document_data import doc_macro
from word_energy.tracking import RunContract,log_contract,tracking_uri
BASE='b37da8076fd54f4da389082278e9733a';EXP='word-energy-wikitext2-joint-context'

def windows(ids,docs,pairs,positions,pair_docs,n,seed):
    ids=np.asarray(ids,dtype=np.int32);docs=np.asarray(docs,dtype=np.int32);positions=np.asarray(positions,dtype=np.int32)
    center=positions[:,0];first=positions[:,1]
    if not np.array_equal(ids[positions],pairs) or not np.array_equal(docs[center],pair_docs):raise ValueError('source occurrence mismatch')
    offsets=np.array([-2,-1,1,2],np.int32);choices=center[:,None]+offsets[None,:];valid=(choices>=0)&(choices<len(ids));clip=np.clip(choices,0,len(ids)-1)
    valid &= (ids[clip]>=0)&(docs[clip]==docs[center,None])&(clip!=first[:,None])
    interior=np.clip(center[:,None]+np.sign(offsets)[None,:],0,len(ids)-1);valid &= ids[interior]>=0
    eligible=np.flatnonzero(valid.any(axis=1))
    if not len(eligible):raise ValueError('no eligible three-word windows')
    rng=np.random.default_rng(seed);selected=rng.choice(eligible,size=n,replace=True);mask=valid[selected]
    choose=(rng.random(n)*mask.sum(axis=1)).astype(np.int32);j=np.argmax(np.cumsum(mask,axis=1)>choose[:,None],axis=1)
    second=choices[selected,j];p=np.column_stack((center[selected],first[selected],second)).astype(np.int32)
    return ids[p],p,docs[center[selected]],selected.astype(np.int32),int(len(eligible))

def build():
    mlflow.set_tracking_uri(tracking_uri())
    base,m=base_load(BASE);vocab={w:i for i,w in enumerate(m['vocabulary'])};assembled={};stats={}
    for split,n,seed in [('train',200000,23101),('validation',10000,23102)]:
        raw,doc=corpus_data(parse(split),vocab);ids=np.asarray(raw,dtype=np.int32)
        words,pos,document,source_rows,eligible=windows(ids,doc,base[f'{split}_pairs'],base[f'{split}_positions'],base[f'{split}_document'],n,seed)
        assembled[f'{split}_windows']=words;assembled[f'{split}_positions']=pos;assembled[f'{split}_document']=document;assembled[f'{split}_source_rows']=source_rows
        stats[split]={'source_pairs':len(base[f'{split}_pairs']),'eligible_source_rows':eligible,'samples':len(words),'unique_occurrence_windows':int(len(np.unique(pos,axis=0))),'documents_sampled':int(len(np.unique(document)))}
    q=base['noise_q'];train=assembled['train_windows'];val=assembled['validation_windows'];counts=base['counts']
    assembled['train_noise']=np.stack([noise_rows(q,train[:,0],5,np.random.default_rng(23110+e)) for e in range(3)])
    assembled['validation_noise']=noise_rows(q,val[:,0],20,np.random.default_rng(23103))
    assembled['orders']=np.stack([np.random.default_rng(23120+e).permutation(len(train)).astype(np.int32) for e in range(3)])
    perm=np.random.default_rng(23130).permutation(len(train)).astype(np.int32);shuffled=train.copy();shuffled[:,1:]=train[perm,1:]
    if not np.array_equal(np.sort(perm),np.arange(len(train))) or not np.array_equal(shuffled[:,0],train[:,0]):raise ValueError('invalid bundle shuffle')
    assembled['shuffle_order']=perm;assembled['shuffled_windows']=shuffled;assembled['counts']=counts;assembled['noise_q']=q
    prior=.25*np.log(counts.astype(float));vp=prior[val[:,0]];vn=prior[assembled['validation_noise']];baseline=pair_metrics(vp,vn,counts[val[:,0]]);baseline['document_macro_auc']=doc_macro(vp,vn,assembled['validation_document'])
    meta={'base_dataset_run':BASE,'base_hashes':m['hashes'],'source_sha256':m['source_sha256'],'vocabulary':m['vocabulary'],'sample_stats':stats,
      'seeds':{'train':23101,'validation':23102,'validation_noise':23103,'training_noise_start':23110,'orders_start':23120,'shuffle':23130},
      'shuffle_fixed_indices':int(np.sum(perm==np.arange(len(perm)))),'shuffle_identical_bundle_rows':int(np.sum(np.all(shuffled[:,1:]==train[:,1:],axis=1))),
      'train_exact_noise_collisions':[int(np.sum(x==train[:,0,None])) for x in assembled['train_noise']],'validation_exact_noise_collisions':int(np.sum(assembled['validation_noise']==val[:,0,None])),
      'frequency_only':baseline,'hashes':{k:sha_array(v) for k,v in assembled.items()},
      'limitations':'posthoc WikiText development; reused original sampled pair occurrences, one shuffle; no sense labels/test/THRML/hardware'}
    return assembled,meta

def load(run_id):
    mlflow.set_tracking_uri(tracking_uri())
    c=MlflowClient();r=c.get_run(run_id)
    if r.info.status!='FINISHED' or r.data.tags.get('model_family')!='wikitext2_joint_context_data':raise ValueError('bad data run')
    root=artifact_dir(c,run_id);meta=json.loads((root/'config.json').read_text(encoding='utf8'))
    with np.load(root/'dataset'/'samples.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    if {k:sha_array(v) for k,v in a.items()}!=meta['hashes']:raise ValueError('array hash mismatch')
    _,old=base_load(BASE)
    if old['hashes']!=meta['base_hashes']:raise ValueError('base data mismatch')
    return a,meta

if __name__=='__main__':
    mlflow.set_tracking_uri(tracking_uri());mlflow.set_experiment(EXP);a,m=build()
    h=hashlib.sha256(''.join(m['source_sha256'].values()).encode()).hexdigest()
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='wikitext2_joint_context_data',energy_convention='no model; center unigram prior',variable_encoding='word IDs',beta_temperature_convention='not applicable',sampler_schedule='fixed window sampling and context bundle permutation',seed=23101,data_hash=h,data_split='existing WikiText train/validation documents; tests untouched',graph_summary='three words per window',limitations=m['limitations'])
    with mlflow.start_run(run_name='joint-context-window-data') as run:
        log_contract(mlflow,contract,m);tmp=ROOT/'.tmp';tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as d:
            p=Path(d)/'samples.npz';np.savez_compressed(p,**a);mlflow.log_artifact(str(p),'dataset')
        for p in (Path(__file__),ROOT/'scripts/build_wikitext2_document_data.py',ROOT/'notes/WIKITEXT2_JOINT_CONTEXT_PROTOCOL.md'):mlflow.log_artifact(str(p),'source_snapshot')
        print(json.dumps({'run_id':run.info.run_id,'stats':m['sample_stats'],'frequency_only':m['frequency_only'],'collisions':m['validation_exact_noise_collisions']}))
