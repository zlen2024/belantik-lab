"""Fixed four-cue extension of the TWSI shared pair-ranking pilot."""
from collections import defaultdict
from pathlib import Path
import csv,hashlib,io,json,os,re,sys,tempfile,time,zipfile
os.environ.setdefault('JAX_PLATFORMS','cpu')
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.build_wikitext2_joint_context_data import load
from scripts.evaluate_swords_external_candidates import DATA,ISING,load_checkpoint
from scripts.evaluate_full_softmax_wic import VECTORS
from scripts.run_ising_corrected_joint import peak_working_set
from scripts.train_twsi_pair_pilot import metrics
from word_energy.ising.exact_training import _imports
from word_energy.shared_multi_context import shared_multi_scores,vector_multi_scores
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract,log_contract,sha256_file,tracking_uri

SPLIT='16e0bf36c0574dcb92cd88cdf1a9b1d3';EXPECTED='0198abbe7f39ac6b5aa67ed6caf389795ad684cd9a2e0339ecc9e3640c55545b'

def build_contexts(archive,vocab,examples,sentence_hashes):
    n=len(vocab);lookup=defaultdict(set)
    with zipfile.ZipFile(archive) as z,z.open('alasca_datasets/twsi_train.tsv') as raw:
        for row in csv.reader(io.TextIOWrapper(raw,encoding='utf8'),delimiter='\t',quoting=csv.QUOTE_NONE):
            if len(row)!=6:raise ValueError('bad TSV')
            try:i=int(row[2])
            except ValueError:continue
            words=row[3].split()
            if not 0<=i<len(words):continue
            target=words[i].lower()
            if not re.fullmatch('[a-z]+',target) or target not in vocab:continue
            ids=[]
            for offset in (-1,1,-2,2):
                j=i+offset
                if 0<=j<len(words):
                    token=words[j].lower()
                    if re.fullmatch('[a-z]+',token) and token in vocab:ids.append(vocab[token])
            if not ids:continue
            oldcue=next((vocab[words[j].lower()] for j in (i-1,i+1) if 0<=j<len(words) and re.fullmatch('[a-z]+',words[j].lower()) and words[j].lower() in vocab),None)
            if oldcue is None:continue
            senthash=hashlib.sha256(' '.join(w.casefold() for w in words).encode()).hexdigest()
            lookup[(senthash,vocab[target],oldcue)].add(tuple((ids+[n]*4)[:4]))
    ctx=[]
    for row,h in zip(examples,sentence_hashes):
        vals=lookup[(str(h),int(row[0]),int(row[1]))]
        if len(vals)!=1:raise ValueError('unmatched or ambiguous context mapping')
        v=next(iter(vals))
        if v[0]!=row[1]:raise ValueError('old cue alignment failure')
        ctx.append(v)
    return np.asarray(ctx,dtype=np.int32)

def main():
    started=time.monotonic();mlflow.set_tracking_uri(tracking_uri());client=MlflowClient()
    archive=ROOT/'data/alasca/alasca_dataset.zip'
    if sha256_file(archive)!=EXPECTED:raise ValueError('archive hash')
    root=artifact_dir(client,SPLIT);conf=json.loads((root/'config.json').read_text(encoding='utf8'))
    with np.load(root/'dataset/split.npz',allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    if {k:sha_array(v) for k,v in arrays.items()}!=conf['hashes']:raise ValueError('split hash')
    _,meta=load(DATA);vocab={w:i for i,w in enumerate(meta['vocabulary'])};n=len(vocab)
    orig=arrays['examples'];contexts=build_contexts(archive,vocab,orig,arrays['sentence_hashes'])
    windows=np.column_stack((orig[:,0],contexts,orig[:,2:4])).astype(np.int32)
    if np.any(windows<0) or np.any(windows[:,[0,5,6]]>=n) or np.any(windows[:,1:5]>n) or np.any(windows[:,5]==windows[:,6]) or not set(np.unique(orig[:,4])).issubset({0,1}):raise ValueError('invalid input')
    tr,va=arrays['train_indices'],arrays['validation_indices'];shuffle=np.random.default_rng(50777).permutation(len(va))
    jax,jnp,optax=_imports()
    if jax.default_backend()!='cpu':raise ValueError('CPU required')
    mlflow.set_experiment('word-energy-natural-context-evaluation')
    limits='repeated TWSI development weak labels; four local slots can miss meaning; no doc disjointness, Gaussian, THRML or chip'
    contract=RunContract('classical','cpu_gpu_simulator','four-cue shared Ising vs contextual vector','higher correlation or vector score','4-spin or tied 10D vector','Ising beta=1','exact sums and Adam',50700,conf['hashes']['examples'],'reused TWSI train connected split','28 spins, 66 edges, max degree 9',limits)
    config={'split_run':SPLIT,'previous_pilot':'dc05b25eaf7c4583b4478bc734a27ebe','split_hashes':conf['hashes'],'contexts_sha256':sha_array(contexts),'vocabulary_sha256':hashlib.sha256(json.dumps(meta['vocabulary'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),'seeds':[17,29,43],'epochs':5,'batch':128,'lr':.003,'temperature':.2,'order_seed_base':50700,'source_runs':{'ising':ISING,'vector':VECTORS},'nonpad_slots':{str(k):int(np.sum(np.sum(contexts!=n,axis=1)==k)) for k in range(5)},'limitations':limits}
    with mlflow.start_run(run_name='twsi-four-cue-pair-ranking-pilot') as run:
        log_contract(mlflow,contract,config);results={};peak=0
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as d:
            fp=Path(d)/'contexts.npz';np.savez_compressed(fp,contexts=contexts);mlflow.log_artifact(str(fp),'dataset')
            for seed in (17,29,43):
                for arm,runs in [('ising',ISING),('vector',VECTORS)]:
                    params={k:jnp.asarray(v) for k,v in load_checkpoint(client,runs[seed],meta).items()}
                    score=jax.jit(shared_multi_scores if arm=='ising' else vector_multi_scores)
                    def loss(p,w,y):
                        s=score(p,w);return jnp.mean(jax.nn.softplus(-jnp.where(y,1.,-1.)*(s[:,0]-s[:,1])/.2))
                    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adam(.003));state=opt.init(params)
                    @jax.jit
                    def step(p,s,w,y):
                        l,g=jax.value_and_grad(loss)(p,w,y);u,s=opt.update(g,s,p);return optax.apply_updates(p,u),s,l
                    def evaluate(p,ix,mode='real'):
                        w=windows[ix].copy()
                        if mode=='removed':w[:,1:5]=n
                        elif mode=='shuffled':w[:,1:5]=windows[ix[shuffle],1:5]
                        out=np.asarray(score(p,jnp.asarray(w)))
                        if not np.isfinite(out).all():raise FloatingPointError('nonfinite scores')
                        return metrics(out,orig[ix])
                    res={'initial_validation':evaluate(params,va),'epochs':[]}
                    for epoch in range(5):
                        order=np.random.default_rng(50700+epoch).permutation(len(tr));losses=[]
                        for start in range(0,len(order),128):
                            ix=tr[order[start:start+128]]
                            params,state,l=step(params,state,jnp.asarray(windows[ix]),jnp.asarray(orig[ix,4],dtype=bool));lv=float(l)
                            if not np.isfinite(lv):raise FloatingPointError('loss')
                            losses.append(lv)
                        peak=max(peak,peak_working_set())
                        if peak>2*1024**3:raise MemoryError('2 GiB cap')
                        res['epochs'].append({'train_mean_batch_loss':float(np.mean(losses))})
                    if any(not np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(params)):raise FloatingPointError('parameters')
                    res.update({'train':evaluate(params,tr),'validation':evaluate(params,va),'validation_context_removed':evaluate(params,va,'removed'),'validation_context_shuffled':evaluate(params,va,'shuffled')})
                    fp=Path(d)/f'{arm}-seed{seed}.npz';np.savez_compressed(fp,**{k:np.asarray(v) for k,v in params.items()});res['checkpoint_sha256']=sha256_file(fp);mlflow.log_artifact(str(fp),'checkpoints')
                    results[f'{arm}-{seed}']=res
                    for stage in ('validation','validation_context_removed','validation_context_shuffled'):
                        for metric in ('pair_accuracy','bidirectional_all_strict_accuracy'):
                            mlflow.log_metric(f'{arm}-{seed}.{stage}.{metric}',res[stage][metric])
                    print(json.dumps({'arm':arm,'seed':seed,'validation':res['validation']}),flush=True)
            config.update({'results':results,'elapsed_seconds':time.monotonic()-started,'peak_working_set_mib':peak/1024**2,'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','JAX_PLATFORMS')}})
            log_contract(mlflow,contract,config);mlflow.log_dict(config,'summary.json')
            for p in (Path(__file__),ROOT/'notes/TWSI_FOUR_CUE_PILOT_PROTOCOL.md',ROOT/'word_energy/shared_multi_context.py',ROOT/'tests/test_shared_multi_context.py'):mlflow.log_artifact(str(p),'source_snapshot')
        print('RUN',run.info.run_id,flush=True)
if __name__=='__main__':main()
