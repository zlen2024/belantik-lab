"""Six-spin checkpoint crossover on SemCor inner-development documents only."""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import hashlib, json, sys, time, zipfile
import mlflow
import numpy as np
from scipy.special import expit, softmax
from mlflow.tracking import MlflowClient
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.evaluate_semcor_grouped_baselines import ARCHIVE, VOCAB_RUN, collect, key
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri

RUNS={17:('5e1958ea3c04411bb246a1dababecf9e','d410f28ca0984d16ac71c4914600974d'),29:('1274d6ea2d8341f1b8a972d5df7b54fb','10ccf64f2c544a25b983cc50ab1ea5d3'),43:('da099306ecff4d4a9f5441f6d2eebcea','828a4c17fe694f9e9bd41f7a6c63d37f')}
STATES=np.array([[2*((n>>i)&1)-1 for i in range(6)] for n in range(64)],dtype=np.float64)
PRODUCTS=np.stack([STATES[:,i]*STATES[:,j] for i,j in combinations(range(6),2)],axis=1)

def q_unary(p):
    return softmax((2.5*np.tanh(p['biases']))@STATES.T+(2*np.tanh(p['internal']))@PRODUCTS.T,axis=1)

def posterior(q,relative,lam,target,examples,cache):
    out=np.broadcast_to(q[target],(len(examples),64)).copy()
    if lam==0:
        return out
    for k in (2,3,4):
        indices=[i for i,x in enumerate(examples) if len(x['cues'])==k]
        if not indices:continue
        if (lam,k) not in cache:
            kernel=np.exp((STATES*(lam*relative/k))@STATES.T)
            cache[lam,k]=q@kernel.T
        messages=cache[lam,k]
        ids=np.array([examples[i]['cues'] for i in indices],dtype=int)
        out[indices]*=np.prod(messages[ids],axis=1)
    out/=out.sum(axis=1,keepdims=True)
    if not np.isfinite(out).all():raise ValueError('posterior nonfinite')
    return out

def main():
    start=time.monotonic();mlflow.set_tracking_uri(tracking_uri());client=MlflowClient()
    baseline=json.loads((artifact_dir(client,'bd4c107ad9654a1288761a207dcd7d98')/'results.json').read_text())
    meta=json.loads((artifact_dir(client,VOCAB_RUN)/'config.json').read_text()); words=meta['vocabulary']
    assert hashlib.sha256(json.dumps(words,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()==baseline['vocabulary_sha256']
    path=ROOT/'data/semcor/semcor.zip';assert sha256_file(path)==ARCHIVE
    vocab={w:i for i,w in enumerate(words)}
    with zipfile.ZipFile(path) as z:
        names=[x for x in z.namelist() if '/tagfiles/' in x and x.endswith('.xml')];assert len(names)==352 and len({Path(x).name for x in names})==352
        outer=sorted(names,key=lambda x:key('semcor-fit-v1:',x))[:282];inner=sorted(outer,key=lambda x:key('semcor-inner-v1:',x))
        assert [key('semcor-inner-v1:',x) for x in inner[:225]]==baseline['inner_fit_document_hashes']
        assert [key('semcor-inner-v1:',x) for x in inner[225:]]==baseline['inner_validation_document_hashes']
        train=collect(z,inner[:225],vocab,set(words[:30])); val=collect(z,inner[225:],vocab,set(words[:30]))
    targets={};rows=[]
    for word,senses in sorted(train.items()):
        qual=[(s,xs) for s,xs in senses.items() if len(xs)>=5 and len({x['doc'] for x in xs})>=3]
        if len(qual)<2:continue
        chosen=dict(sorted(qual,key=lambda x:(-len(x[1]),x[0]))[:2]);labels=sorted(chosen)
        examples=[(s,x) for s in labels for x in val.get(word,{}).get(s,[])]
        if not examples:continue
        begin=len(rows)
        for sense,x in examples:rows.append({'word':word,'sense':sense,'doc':hashlib.sha256(('semcor-doc-v1:'+x['doc']).encode()).hexdigest(),'cue_count':len(x['cues'])})
        targets[word]={'labels':labels,'train':chosen,'examples':[x for _,x in examples],'indices':np.arange(begin,len(rows))}
    assert len(rows)==baseline['selected_validation_occurrences']==1279 and len(targets)==baseline['validation_targets']==146
    base_targets={r['word']:r for r in baseline['per_target']}
    for word,t in targets.items():
        actual=Counter(rows[i]['sense'] for i in t['indices']);assert dict(actual)==base_targets[word]['validation_counts'] and t['labels']==base_targets[word]['senses']
    balanced_words=[w for w,t in targets.items() if len({rows[i]['sense'] for i in t['indices']})==2];assert len(balanced_words)==94
    checkpoints={}; strengths={}
    for seed,ids in RUNS.items():
        for arm,rid in zip(('fixed','learned'),ids):
            run=client.get_run(rid);root=artifact_dir(client,rid);cfg=json.loads((root/'config.json').read_text());cp=root/'results/checkpoint.npz'
            if run.info.status!='FINISHED' or cfg['arm']!=arm or cfg['seed']!=seed or cfg['data_run']!='3387fc4fb99646f1b7aacd518f39bf10' or sha256_file(cp)!=cfg['checkpoint_sha256']:raise ValueError('checkpoint provenance')
            with np.load(cp,allow_pickle=False) as z:p={k:z[k] for k in z.files}
            checkpoints[seed,arm]=(q_unary(p),softmax(p['interaction']),float(1.5*expit(p['strength'])))
        strengths[seed]=checkpoints[seed,'learned'][2]
    correct={}; all_pred={}
    for seed in RUNS:
        for arm in ('fixed','learned'):
            q,r,native=checkpoints[seed,arm];cache={}
            tests={'zero':0.,'point8':.8,'learned_strength':strengths[seed]}
            for condition in tests:all_pred[f'{seed}:{arm}:{condition}']=np.zeros(len(rows),dtype=np.int8)
            for word,t in targets.items():
                labels=t['labels'];target=vocab[word]
                means=np.stack([posterior(q,r,native,target,t['train'][label],cache).mean(axis=0) for label in labels])
                for condition,lam in tests.items():
                    p=posterior(q,r,lam,target,t['examples'],cache)
                    if lam==0:np.testing.assert_allclose(p,np.broadcast_to(q[target],p.shape),atol=1e-14,rtol=0)
                    distance=((p[:,None,:]-means[None,:,:])**2).sum(axis=2);pred=np.argmin(distance,axis=1)
                    for i,y in zip(t['indices'],pred):all_pred[f'{seed}:{arm}:{condition}'][i]=int(rows[i]['sense']==labels[y])
    word_idx={w:targets[w]['indices'] for w in balanced_words}; y=np.array([r['sense'] for r in rows],dtype=object)
    def balanced(pred,idx=None):
        vals=[]
        if idx is None:idx=np.arange(len(rows))
        weight=np.bincount(idx,minlength=len(rows))
        for w in balanced_words:
            ix=word_idx[w]; labels=targets[w]['labels'];scores=[]
            for label in labels:
                selected=ix[y[ix]==label];total=weight[selected].sum()
                if not total:break
                scores.append(float(np.dot(weight[selected],pred[selected])/total))
            if len(scores)==2:vals.append(sum(scores)/2)
        return float(np.mean(vals)) if vals else float('nan'),len(vals)
    summary={k:{'target_macro_balanced':balanced(v)[0],'accuracy':float(v.mean()),'cue_count_accuracy':{str(n):float(v[[i for i,r in enumerate(rows) if r['cue_count']==n]].mean()) for n in (2,3,4)}} for k,v in all_pred.items()}
    docs=sorted({r['doc'] for r in rows});bydoc={d:np.array([i for i,r in enumerate(rows) if r['doc']==d]) for d in docs};assert len(docs)==31
    rng=np.random.default_rng(20261122);boot={f'{seed}:{effect}':[] for seed in RUNS for effect in ('native_minus_fixed_native','native_minus_fixed_scaled')};support=[]
    for _ in range(2000):
        idx=np.concatenate([bydoc[d] for d in rng.choice(docs,size=len(docs),replace=True)])
        for seed in RUNS:
            learned,n=balanced(all_pred[f'{seed}:learned:learned_strength'],idx)
            for effect,control in [('native_minus_fixed_native','point8'),('native_minus_fixed_scaled','learned_strength')]:
                base,_=balanced(all_pred[f'{seed}:fixed:{control}'],idx);boot[f'{seed}:{effect}'].append(learned-base)
        support.append(n)
    effects={k:{'point':summary[k.split(':')[0]+':learned:learned_strength']['target_macro_balanced']-summary[k.split(':')[0]+':fixed:'+('point8' if k.endswith('native_minus_fixed_native') else 'learned_strength')]['target_macro_balanced'],'bootstrap_95':np.quantile(v,[.025,.975]).tolist()} for k,v in boot.items()}
    result={'source_baseline':'bd4c107ad9654a1288761a207dcd7d98','source_runs':RUNS,'native_learned_strengths':strengths,'scored_rows':len(rows),'scored_documents':len(docs),'validation_targets':len(targets),'balanced_targets':len(balanced_words),'summary':summary,'paired_effects':effects,'bootstrap_replicates':2000,'bootstrap_target_support_min_max':[min(support),max(support)],'majority_baseline':baseline['majority'],'context_nb_baseline':baseline['cues'],'elapsed_seconds':time.monotonic()-start,'limitations':'repeated development; supervised native-strength centroids; two selected senses only; no outer reserve, THRML, Gaussian or hardware'}
    mlflow.set_experiment('word-energy-learnable-w-strength')
    with mlflow.start_run(run_name='semcor-inner-dev-crossover') as run:
        log_contract(mlflow,RunContract('classical','cpu_gpu_simulator','ising6 shared-center supervised sense probe','exact normalized 64-state posterior, lower squared centroid distance','spins {-1,+1}','beta=1','exact star enumeration',20261122,ARCHIVE,'225 fit, 57 inner validation; outer 70 unopened','up to 30 spins, diagonal spokes',result['limitations']),result)
        mlflow.log_dict(result,'results.json');mlflow.log_dict([dict(r,correct={k:int(v[i]) for k,v in all_pred.items()}) for i,r in enumerate(rows)],'paired_rows.json')
        for k,v in summary.items():mlflow.log_metric(k.replace(':','.')+'.balanced',v['target_macro_balanced'])
        for p in (Path(__file__),ROOT/'notes/LEARNABLE_W_SEMCOR_PROTOCOL.md'):mlflow.log_artifact(str(p),'source_snapshot')
        print('RUN',run.info.run_id,json.dumps(result),flush=True)
if __name__=='__main__':main()
