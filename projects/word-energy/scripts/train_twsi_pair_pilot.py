"""Exploratory matched shared-Ising/vector weak pair-ranking pilot."""
from pathlib import Path
import hashlib, json, os, sys, tempfile, time
os.environ.setdefault('JAX_PLATFORMS','cpu')
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.build_wikitext2_joint_context_data import load
from scripts.evaluate_swords_external_candidates import DATA, ISING, load_checkpoint
from scripts.evaluate_full_softmax_wic import VECTORS
from scripts.run_ising_corrected_joint import peak_working_set
from word_energy.ising.exact_training import _imports
from word_energy.shared_rank_training import shared_pair_scores, paired_rank_loss, validate_training_batch
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract,log_contract,tracking_uri

SPLIT='16e0bf36c0574dcb92cd88cdf1a9b1d3'
SEEDS=(17,29,43)
def vector_scores(p,w):
    _,jnp,_=_imports();v=p['shared'];t,c,a,b=(v[w[:,i]] for i in range(4))
    ref=t+c+t*c
    return jnp.stack((jnp.sum(ref*a,axis=1),jnp.sum(ref*b,axis=1)),axis=1)

def metrics(scores,rows):
    delta=scores[:,0]-scores[:,1];sign=2*rows[:,4]-1
    credit=(delta*sign>0).astype(float)+.5*(delta==0)
    targets={};groups={}
    for i,r in enumerate(rows):
        targets.setdefault(int(r[0]),[]).append(i)
        groups.setdefault((int(r[0]),int(r[2]),int(r[3])),[]).append(i)
    bidirectional=[idx for idx in groups.values() if len(set(rows[idx,4]))==2]
    return {'pair_accuracy':float(credit.mean()),'target_macro_accuracy':float(np.mean([credit[x].mean() for x in targets.values()])),
      'pair_group_macro_accuracy':float(np.mean([credit[x].mean() for x in groups.values()])),
      'bidirectional_all_strict_accuracy':float(np.mean([np.all(delta[x]*sign[x]>0) for x in bidirectional])) if bidirectional else None,
      'bidirectional_groups':len(bidirectional),'ties':int(np.sum(delta==0)),'examples':len(rows),'target_count':len(targets),'group_count':len(groups)}

def main():
    started=time.monotonic();mlflow.set_tracking_uri(tracking_uri());client=MlflowClient()
    root=artifact_dir(client,SPLIT);conf=json.loads((root/'config.json').read_text(encoding='utf8'))
    with np.load(root/'dataset/split.npz',allow_pickle=False) as f: arrays={k:f[k] for k in f.files}
    if {k:sha_array(v) for k,v in arrays.items()}!=conf['hashes']:raise ValueError('split hash mismatch')
    examples=arrays['examples'];train=examples[arrays['train_indices']];val=examples[arrays['validation_indices']]
    shuffled=val.copy();shuffled[:,1]=val[np.random.default_rng(50777).permutation(len(val)),1]
    _,meta=load(DATA);n=len(meta['vocabulary'])
    for rows in (train,val):validate_training_batch(rows[:,:4],rows[:,4].astype(bool),.2,n)
    if not set(np.unique(examples[:,4])).issubset({0,1}):raise ValueError('invalid labels')
    jax,jnp,optax=_imports()
    if jax.default_backend()!='cpu':raise ValueError('requires CPU')
    mlflow.set_experiment('word-energy-natural-context-evaluation')
    with mlflow.start_run(run_name='twsi-shared-pair-ranking-pilot') as run:
        limitations='reused TWSI train weak labels, single cue, no source-independent or test claims; no Gaussian/THRML/hardware'
        contract=RunContract('classical','cpu_gpu_simulator','shared Ising vs contextual vector ranking','higher raw correlation or vector score','4 spins {-1,+1} or 10D vector','Ising beta=1','exact sums, Adam',50700,conf['hashes']['examples'],'exploratory connected split of TWSI train','16 Ising spins, max degree 6',limitations)
        vocabulary_sha=hashlib.sha256(json.dumps(meta['vocabulary'],ensure_ascii=False,separators=(',',':')).encode('utf8')).hexdigest()
        configuration={'split_run':SPLIT,'split_hashes':conf['hashes'],'vocabulary_sha256':vocabulary_sha,'seeds':SEEDS,'epochs':5,'batch':128,'lr':.003,'temperature':.2,'order_seed_base':50700,'source_runs':{'ising':ISING,'vector':VECTORS},'limits':limitations}
        log_contract(mlflow,contract,configuration)
        results={}
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as d:
            for seed in SEEDS:
                for arm,runs in [('ising',ISING),('vector',VECTORS)]:
                    params={k:jnp.asarray(v) for k,v in load_checkpoint(client,runs[seed],meta).items()}
                    if arm=='ising':
                        score=jax.jit(shared_pair_scores)
                        loss=lambda p,w,y:paired_rank_loss(p,w,y,.2)
                    else:
                        score=jax.jit(vector_scores)
                        loss=lambda p,w,y:jnp.mean(jax.nn.softplus(-jnp.where(y,1.,-1.)*(vector_scores(p,w)[:,0]-vector_scores(p,w)[:,1])/.2))
                    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adam(.003));state=opt.init(params)
                    @jax.jit
                    def step(p,s,w,y):
                        l,g=jax.value_and_grad(loss)(p,w,y);u,s=opt.update(g,s,p);return optax.apply_updates(p,u),s,l
                    def evaluate(p,rows,removed=False):
                        w=rows[:,:4].copy()
                        if removed:
                            ext=dict(p);key='biases' if arm=='ising' else 'shared';dummy=len(p[key]);w[:,1]=dummy
                            ext[key]=jnp.concatenate((p[key],jnp.zeros((1,p[key].shape[1]))),axis=0)
                            if arm=='ising':ext['internal']=jnp.concatenate((p['internal'],jnp.zeros((1,6))),axis=0)
                            p=ext
                        scores=np.asarray(score(p,jnp.asarray(w)))
                        if not np.isfinite(scores).all():raise FloatingPointError('nonfinite score')
                        return metrics(scores,rows)
                    outcome={'initial_validation':evaluate(params,val),'epochs':[]}
                    for epoch in range(5):
                        order=np.random.default_rng(50700+epoch).permutation(len(train));losses=[]
                        for start in range(0,len(order),128):
                            batch=train[order[start:start+128]]
                            params,state,l=step(params,state,jnp.asarray(batch[:,:4]),jnp.asarray(batch[:,4],dtype=bool));lv=float(l)
                            if not np.isfinite(lv):raise FloatingPointError('nonfinite training loss')
                            losses.append(lv)
                        if peak_working_set()>2*1024**3:raise MemoryError('2 GiB working set cap')
                        outcome['epochs'].append({'train_mean_batch_loss':float(np.mean(losses))})
                    if any(not np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(params)):raise FloatingPointError('nonfinite parameters')
                    outcome.update({'train':evaluate(params,train),'validation':evaluate(params,val),'validation_context_removed':evaluate(params,val,True),'validation_context_shuffled':evaluate(params,shuffled)})
                    checkpoint=Path(d)/f'{arm}-seed{seed}.npz';np.savez_compressed(checkpoint,**{k:np.asarray(v) for k,v in params.items()});mlflow.log_artifact(str(checkpoint),'checkpoints')
                    results[f'{arm}-{seed}']=outcome
                    print(json.dumps({'arm':arm,'seed':seed,'validation':outcome['validation']}),flush=True)
            configuration['results']=results;configuration['elapsed_seconds']=time.monotonic()-started
            log_contract(mlflow,contract,configuration);mlflow.log_dict(configuration,'summary.json')
            for f in (Path(__file__),ROOT/'notes/TWSI_PAIR_RANK_PILOT_PROTOCOL.md',ROOT/'word_energy/shared_rank_training.py'):
                mlflow.log_artifact(str(f),'source_snapshot')
        print('RUN',run.info.run_id,flush=True)
if __name__=='__main__':main()
