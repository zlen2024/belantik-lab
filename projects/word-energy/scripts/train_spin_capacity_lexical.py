"""One seed/arm of matched exact-Ising and vector lexical capacity pilot."""
from pathlib import Path
import json,os,sys,tempfile,time
os.environ.setdefault('JAX_PLATFORMS','cpu')
import mlflow
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_wikitext2_joint_context_data import load
from scripts.build_wikitext2_document_data import doc_macro
from scripts.run_ising_corrected_joint import peak_working_set
from word_energy.ising.exact_training import _imports
from word_energy.spin_capacity_lexical import initial,scores
from word_energy.text8_lexical_models import pair_metrics
from word_energy.tracking import RunContract,log_contract,sha256_file,tracking_uri

DATA='f1b7db5231ed4a5788b566e388f8e731';ARMS=('ising4','ising6','vector10','vector21');SEEDS=(17,29,43)
def main(arm,seed):
    if arm not in ARMS or seed not in SEEDS:raise ValueError('outside panel')
    start=time.monotonic();mlflow.set_tracking_uri(tracking_uri());a,meta=load(DATA);jax,jnp,optax=_imports()
    if jax.default_backend()!='cpu':raise ValueError('CPU required')
    ntrain=10000;nval=2000;counts=a['counts'];prior=.25*np.log(counts.astype(float))
    params=initial(arm,seed,len(counts));opt=optax.chain(optax.clip_by_global_norm(5.),optax.adam(.01));state=opt.init(params)
    def loss(p,pos,neg):
        b=pos.shape[0];m=neg.shape[1];bad=jnp.broadcast_to(pos[:,None,:],(b,m,2)).at[:,:,0].set(neg).reshape((-1,2))
        good=scores(p,pos,prior,arm);false=scores(p,bad,prior,arm).reshape((b,m))
        return jnp.mean(jax.nn.softplus(1-good[:,None]+false))
    @jax.jit
    def step(p,s,pos,neg):
        value,gradient=jax.value_and_grad(loss)(p,pos,neg);u,s=opt.update(gradient,s,p);return optax.apply_updates(p,u),s,value
    score=jax.jit(lambda p,x:scores(p,x,prior,arm))
    windows=a['validation_windows'][:nval,:2];noise=a['validation_noise'][:nval,:5];docs=a['validation_document'][:nval]
    def evaluate(p):
        positives=[];negatives=[]
        for i in range(0,nval,128):
            w=windows[i:i+128];z=noise[i:i+128];bad=np.broadcast_to(w[:,None,:],(len(w),z.shape[1],2)).copy();bad[:,:,0]=z
            positives.append(np.asarray(score(p,w)));negatives.append(np.asarray(score(p,bad.reshape(-1,2))).reshape(len(w),-1))
        pos=np.concatenate(positives);neg=np.concatenate(negatives);m=pair_metrics(pos,neg,counts[windows[:,0]])
        m['document_macro_auc']=doc_macro(pos,neg,docs);return m,pos,neg
    if not arm.startswith('vector'):
        w=windows[:100];zero=np.asarray(scores(params,w,prior,arm,coupling=False))
        np.testing.assert_allclose(zero,prior[w[:,0]],atol=1e-12,rtol=0)
    n=int(arm[5:]) if arm.startswith('ising') else 0
    family='exact normalized Ising pair' if n else 'shared vector pair'
    graph=f'{2*n} spins, {n*(n-1)+n} edges, max degree {n}' if n else f'{arm[6:]}-dimensional shared vector'
    limits='reused WikiText train/dev, first 10k/2k windows and 5 noise; offsets on host; pairwise lexical only; exact classical, no sense or hardware claims'
    contract=RunContract('classical','cpu_gpu_simulator',family,'higher normalized log-partition or dot plus 0.25 log count','spins {-1,+1}' if n else 'real vectors','beta=1' if n else 'not applicable','exact pair summation and Adam',seed,meta['source_sha256']['train'],'WikiText train/development; official test closed',graph,limits)
    config={'arm':arm,'seed':seed,'source_run':DATA,'dataset_hashes':meta['hashes'],'train_rows':ntrain,'validation_rows':nval,'negatives':5,'epochs':3,'batch':128,'lr':.01,'clip':5.,'prior_scale':.25,'normalization':'Ising g_i=1.5 sigmoid(raw_i)/d; same maximum total g', 'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','JAX_PLATFORMS')},'limitations':limits}
    mlflow.set_experiment('word-energy-spin-capacity-lexical')
    with mlflow.start_run(run_name=f'{arm}-seed{seed}') as run:
        log_contract(mlflow,contract,config);mlflow.log_params({'arm':arm,'seed':seed,'train_rows':ntrain,'validation_rows':nval,'epochs':3,'batch':128})
        val0,_,_=evaluate(params);config['initial_validation']=val0
        if n:
            initial_g=1.5*expit(np.asarray(params['interaction']))/n
            config['initial_couplings']=initial_g.tolist();mlflow.log_metric('coupling_sum',float(initial_g.sum()),step=0)
        freq=pair_metrics(prior[windows[:,0]],prior[noise],counts[windows[:,0]])
        freq['document_macro_auc']=doc_macro(prior[windows[:,0]],prior[noise],docs);config['frequency']=freq
        for epoch in range(3):
            order=a['orders'][epoch];order=order[order<ntrain]
            if len(order)!=ntrain or len(np.unique(order))!=ntrain:raise ValueError('order invalid')
            total=0.;seen=0
            for i in range(0,len(order),128):
                idx=order[i:i+128];w=a['train_windows'][idx,:2];neg=a['train_noise'][epoch,idx,:5]
                params,state,value=step(params,state,w,neg);v=float(value)
                if not np.isfinite(v):raise FloatingPointError('nonfinite loss')
                total+=v*len(idx);seen+=len(idx)
            memory=peak_working_set()
            if memory>2*1024**3:raise MemoryError('2 GiB cap')
            if time.monotonic()-start>900:raise TimeoutError('per-run time budget exceeded')
            m,pos,neg=evaluate(params);m.update({'train_loss':total/seen,'peak_working_set_mib':memory/1024**2})
            if n:
                g=1.5*expit(np.asarray(params['interaction']))/n
                m['couplings']=g.tolist();m['coupling_sum']=float(g.sum());mlflow.log_metric('coupling_sum',m['coupling_sum'],step=epoch+1)
            config[f'epoch{epoch+1}']=m
            for k in ('noise_pairwise_auc','document_macro_auc','train_loss'):mlflow.log_metric(k,m[k],step=epoch+1)
            print(json.dumps({'run':run.info.run_id,'arm':arm,'seed':seed,'epoch':epoch+1,'auc':m['noise_pairwise_auc'],'doc_auc':m['document_macro_auc']}),flush=True)
        if any(not np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(params)):raise FloatingPointError('nonfinite params')
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as d:
            ck=Path(d)/'checkpoint.npz';pred=Path(d)/'validation_scores.npz'
            np.savez_compressed(ck,**{k:np.asarray(v) for k,v in params.items()});np.savez_compressed(pred,positive=pos,negative=neg,document=docs)
            config['checkpoint_sha256']=sha256_file(ck);config['prediction_sha256']=sha256_file(pred)
            mlflow.log_artifact(str(ck),'results');mlflow.log_artifact(str(pred),'results')
        config['elapsed_seconds']=time.monotonic()-start;config['run_complete']=True;log_contract(mlflow,contract,config)
        for f in (Path(__file__),ROOT/'word_energy/spin_capacity_lexical.py',ROOT/'tests/test_spin_capacity_lexical.py',ROOT/'notes/SPIN_CAPACITY_LEXICAL_PILOT_PROTOCOL.md'):mlflow.log_artifact(str(f),'source_snapshot')
        print('FINISHED',run.info.run_id,flush=True)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))
