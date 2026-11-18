"""Frozen label-blind WiC context probe of pair-trained WikiText models."""
from pathlib import Path
import json,sys,time
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit,softmax
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.build_wikitext2_document_data import load
from scripts.audit_text8_canonical_evaluator import artifact_dir
from word_energy.text8_development import sha_array
from word_energy.wic_context_probe import STATES,extract,star,negative_js,cosine,auc
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file
DATA='b37da8076fd54f4da389082278e9733a';WIC=ROOT/'data/wic/validation.parquet';SHA='22cc5df62bf76c8ed47ea0c92b4bb1bce698984ac1c10493250365d3419ac6da';SEEDS=(17,29,43)

def model(client,exp_name,arm,seed,stage,mode=None):
    exp=client.get_experiment_by_name(exp_name)
    matches=[r for r in client.search_runs([exp.experiment_id],max_results=1000) if r.data.tags.get('arm')==arm and r.data.tags.get('seed')==str(seed) and (mode is None or r.data.tags.get('mode')==mode)]
    if len(matches)!=1 or matches[0].info.status!='FINISHED':raise ValueError('missing model '+str((arm,seed,mode)))
    run=matches[0];root=artifact_dir(client,run.info.run_id);cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    if not cfg['run_complete'] or cfg['dataset_run_id']!=DATA:raise ValueError('dataset/config mismatch')
    with np.load(root/'checkpoints'/f'{stage}.npz',allow_pickle=False) as z:p={k:z[k] for k in z.files}
    if {k:sha_array(v) for k,v in p.items()}!=cfg['checkpoint_hashes'][stage]:raise ValueError('checkpoint hash mismatch')
    return p,run.info.run_id

def vectors(p,target,ctx,shared,target_conditioned):
    left=p['shared'] if shared else p['input'];right=p['shared'] if shared else p['output'];value=np.mean(right[ctx],axis=0)
    return value+left[target] if target_conditioned else value

def bootstrap(labels,groups,seed_scores):
    labels=np.asarray(labels);groups=np.asarray(groups);targets=np.unique(groups);rng=np.random.default_rng(22150)
    draws=rng.integers(0,len(targets),size=(1000,len(targets)));group_idx=np.searchsorted(targets,groups)
    pos=np.flatnonzero(labels==1);neg=np.flatnonzero(labels==0)
    matrices=[(s[pos,None]>s[None,neg])+.5*(s[pos,None]==s[None,neg]) for s in seed_scores];valid=[]
    for draw in draws:
        w=np.bincount(draw,minlength=len(targets))[group_idx];wp=w[pos];wn=w[neg];denom=wp.sum()*wn.sum()
        if denom:valid.append(float(np.mean([wp@mat@wn/denom for mat in matrices])))
    if not valid:return {'interval':None,'valid_draws':0}
    return {'interval':[float(x) for x in np.percentile(valid,[2.5,97.5])],'valid_draws':len(valid)}

def neighbors(raw,role,vocab,counts):
    if role:bt,kt,bc,kc=raw['target_biases'],raw['target_internal'],raw['context_biases'],raw['context_internal']
    else:bt=bc=raw['biases'];kt=kc=raw['internal']
    u=vocab.index('bank');prod=STATES[:,0]*STATES[:,1];p=softmax(STATES@(2.5*np.tanh(bt[u]))+prod*(2*np.tanh(kt[u])))
    q=softmax((2.5*np.tanh(bc))@STATES.T+(2*np.tanh(kc))[:,None]*prod,axis=1)
    g=1.5*expit(raw['interaction']);K=np.expm1(np.sum(STATES[:,None,:]*STATES[None,:,:]*g,axis=-1))
    raw_score=np.log1p(np.einsum('i,ij,kj->k',p,K,q));output={}
    for mode,values in [('energy_only',raw_score),('with_frequency',raw_score+.25*np.log(counts))]:
        ranks=values.copy();ranks[u]=-np.inf;ix=np.argsort(-ranks,kind='stable')
        wanted={w:{'rank':int(np.where(ix==vocab.index(w))[0][0]+1),'score':float(values[vocab.index(w)])} for w in ('river','money','water','financial')}
        output[mode]={'top5':[{'word':vocab[i],'score':float(values[i])} for i in ix[:5]],'selected':wanted}
    output['bank_posterior']={w:star(raw,u,[vocab.index(w)],role).tolist() for w in ('river','money')}
    return output

def execute():
    client=MlflowClient();a,m=load(DATA)
    if sha256_file(WIC)!=SHA:raise ValueError('WiC source hash mismatch')
    vocab={w:i for i,w in enumerate(m['vocabulary'])};cols=['word','sentence1','sentence2','start1','end1','start2','end2','idx']
    rows=pq.read_table(WIC,columns=cols).to_pylist();selected=[];reasons=Counter()
    for row in rows:
        word=row['word'].lower()
        if word not in vocab:reasons['target_oov']+=1;continue
        c1=extract(row['sentence1'],row['start1'],row['end1'],vocab,stats=True);c2=extract(row['sentence2'],row['start2'],row['end2'],vocab,stats=True)
        if c1 is None or c2 is None:reasons['invalid_span']+=1;continue
        if not c1[0] or not c2[0]:reasons['empty_in_vocab_context']+=1;continue
        selected.append({'idx':row['idx'],'word':word,'target':vocab[word],'context1':c1[0],'context2':c2[0],'context_slots1':c1[1],'context_slots2':c2[1]})
    if not selected:raise ValueError('no eligible examples')
    predictions={};source_runs={};illustrations={};vector_cache={}
    for arm in ('shared3','directed3'):
        for seed in SEEDS:
            p,rid=model(client,'word-energy-wikitext2-matched-vectors',arm,seed,'epoch3')
            vector_cache[(arm,seed)]=p;source_runs[f'{arm}:{seed}']=rid
    safe=[]
    for row in selected:
        try:
            for (arm,seed),p in vector_cache.items():
                for target_cond in (True,False):
                    cosine(vectors(p,row['target'],row['context1'],arm=='shared3',target_cond),vectors(p,row['target'],row['context2'],arm=='shared3',target_cond))
            safe.append(row)
        except ValueError:reasons['zero_or_nonfinite_vector']+=1
    selected=safe
    if not selected:raise ValueError('no common valid examples')
    for arm in ('tied-frequency','role-frequency'):
        role=arm=='role-frequency'
        for mode,stage in [('original','epoch3'),('shuffled','epoch3'),('initial','initial')]:
            series=[];entropies=[]
            for seed in SEEDS:
                p,rid=model(client,'word-energy-wikitext2-doc-development',arm,seed,stage,'shuffled' if mode=='shuffled' else 'original')
                source_runs[f'{arm}:{mode}:{seed}']=rid;value=[];entropy=[]
                for row in selected:
                    x=star(p,row['target'],row['context1'],role);y=star(p,row['target'],row['context2'],role)
                    value.append(negative_js(x,y));entropy.extend([-np.sum(x*np.log(x)),-np.sum(y*np.log(y))])
                series.append(value);entropies.append(float(np.mean(entropy)))
                if mode=='original':illustrations[f'{arm}:{seed}']=neighbors(p,role,m['vocabulary'],a['counts'])
            predictions[f'{arm}:{mode}']={'scores':series,'posterior_entropy_mean':entropies}
    for arm in ('shared3','directed3'):
        shared=arm=='shared3';series=[];context_only=[]
        for seed in SEEDS:
            p=vector_cache[(arm,seed)]
            x=[];y=[]
            for row in selected:
                for target_cond,dest in ((True,x),(False,y)):
                    c1=vectors(p,row['target'],row['context1'],shared,target_cond);c2=vectors(p,row['target'],row['context2'],shared,target_cond)
                    dest.append(cosine(c1,c2))
            series.append(x);context_only.append(y)
        predictions[f'{arm}:target_plus_context']={'scores':series};predictions[f'{arm}:context_only']={'scores':context_only}
    predictions['bag_jaccard']={'scores':[[len(set(r['context1'])&set(r['context2']))/len(set(r['context1'])|set(r['context2'])) for r in selected]]}
    predictions['context_count']={'scores':[[-abs(len(r['context1'])-len(r['context2'])) for r in selected]]}
    # Labels are deliberately read only after eligibility and every score is frozen.
    label_rows=pq.read_table(WIC,columns=['idx','label']).to_pylist();label_map={r['idx']:r['label'] for r in label_rows}
    labels=np.array([label_map[r['idx']] for r in selected],dtype=np.int32)
    if not set(labels.tolist())<={0,1} or len(set(labels.tolist()))!=2:raise ValueError('invalid labels/subset')
    all_labels=np.array([r['label'] for r in label_rows],dtype=np.int32)
    if not set(all_labels.tolist())<={0,1}:raise ValueError('invalid full labels')
    groups=[r['word'] for r in selected];summaries={}
    for key,value in predictions.items():
        scores=[np.asarray(s,dtype=float) for s in value['scores']]
        if not all(np.isfinite(s).all() and len(s)==len(labels) for s in scores):raise ValueError('invalid scores')
        aucs=[auc(labels,s) for s in scores];ties=[float(np.mean(s[labels==1,None]==s[None,labels==0])) for s in scores]
        summaries[key]={'auc_per_seed':aucs,'auc_mean':float(np.mean(aucs)),'tie_fraction':ties,'target_cluster_bootstrap':bootstrap(labels,groups,scores)}
        if 'posterior_entropy_mean' in value:summaries[key]['posterior_entropy_mean']=value['posterior_entropy_mean']
    return {'wic_sha256':SHA,'dataset_run_id':DATA,'dataset_hashes':m['hashes'],'source_runs':source_runs,'total_examples':len(rows),'eligible_examples':len(selected),
      'excluded':dict(reasons),'target_in_vocab_examples':len(rows)-reasons['target_oov'],'all_positive':int(all_labels.sum()),'all_negative':int(len(all_labels)-all_labels.sum()),'excluded_positive':int(all_labels.sum()-labels.sum()),'excluded_negative':int((len(all_labels)-all_labels.sum())-(len(labels)-labels.sum())), 'context_slots':sum(r['context_slots1']+r['context_slots2'] for r in selected),'context_in_vocab':sum(len(r['context1'])+len(r['context2']) for r in selected),'positive_examples':int(labels.sum()),'negative_examples':int(len(labels)-labels.sum()),'distinct_target_words':len(set(groups)),
      'context_lengths':dict(Counter(str((len(r['context1']),len(r['context2']))) for r in selected)),
      'selected':selected,'labels':labels.tolist(),'predictions':predictions,'summaries':summaries,'bank_illustrations':illustrations,
      'limitations':'WiC development only, selected target/context in-vocabulary subset; novel untrained posterior; context/topic controls; CC-BY-NC-4; no benchmark test or hardware'}

if __name__=='__main__':
    start=time.monotonic();mlflow.set_tracking_uri(tracking_uri());out=execute();out['elapsed_seconds']=time.monotonic()-start
    mlflow.set_experiment('word-energy-wic-context-probe')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='wic_context_readout',energy_convention='exact normalized Ising star posterior or vector cosine; no physical energy',variable_encoding='spins {-1,+1} or 3D vectors',beta_temperature_convention='beta=1 for Ising',sampler_schedule='exact four-state latent center, no WiC training',seed=22150,data_hash=SHA,data_split='WiC validation only; all test splits untouched',graph_summary='bounded star; up to four context words',limitations=out['limitations'])
    with mlflow.start_run(run_name='wic-development-context-probe') as run:
        log_contract(mlflow,contract,out)
        for key,v in out['summaries'].items():mlflow.log_metric(key.replace(':','.')+'.auc_mean',v['auc_mean'])
        for p in (Path(__file__),ROOT/'word_energy/wic_context_probe.py',ROOT/'tests/test_wic_context_probe.py',ROOT/'notes/WIC_CONTEXT_PROBE_PROTOCOL.md'):mlflow.log_artifact(str(p),'source_snapshot')
        print(json.dumps({'run_id':run.info.run_id,'eligible':out['eligible_examples'],'total':out['total_examples'],'excluded':out['excluded'],'positive':out['positive_examples'],'negative':out['negative_examples'],'targets':out['distinct_target_words'],'summaries':out['summaries'],'bank':out['bank_illustrations']},indent=2))
