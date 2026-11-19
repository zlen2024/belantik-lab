"""One joint-six-spin or controlled model on fixed WikiText windows."""
from pathlib import Path
import hashlib,json,os,sys,tempfile,time
import numpy as np
from scipy.special import expit,softmax
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.build_wikitext2_joint_context_data import load,EXP
from scripts.build_wikitext2_document_data import doc_macro
from scripts.train_text8_lexical_development import snapshot,log_metrics
from scripts.run_ising_corrected_joint import peak_working_set
from word_energy.ising.exact_training import _imports
from word_energy.wikitext_joint_context import ARMS,SEEDS,initial,functions,score_windows
from word_energy.wikitext_joint_evaluation import canonical_scores
from word_energy.text8_lexical_models import pair_metrics
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract,log_contract,tracking_uri

def validation(params,score,loss,a):
    windows=a['validation_windows'];noise=a['validation_noise'];p,n=canonical_scores(params,score,windows,noise)
    out=pair_metrics(p,n,a['counts'][windows[:,0]]);out['document_macro_auc']=doc_macro(p,n,a['validation_document'])
    total=0.
    for i in range(0,len(windows),500):
        j=min(i+500,len(windows));total+=float(loss(params,windows[i:j],noise[i:j,:5]))*(j-i)
    out['objective_loss']=total/len(windows);return out

def diagnostics(params,a):
    raw={k:np.asarray(v) for k,v in params.items()};w=a['validation_windows'];states=np.array([[-1.,-1.],[1.,-1.],[-1.,1.],[1.,1.]]);prod=states[:,0]*states[:,1]
    b=2.5*np.tanh(raw['biases']);k=2*np.tanh(raw['internal']);g=1.5*expit(raw['interaction'])
    q=softmax(b@states.T+k[:,None]*prod,axis=1);center=q[w[:,0]];cross=np.exp(np.sum(states[:,None,:]*states[None,:,:]*g/2,axis=-1))
    d1=q[w[:,1]]@cross.T;d2=q[w[:,2]]@cross.T;post=center*d1*d2;post/=post.sum(axis=1,keepdims=True)
    joint=np.log(np.sum(center*d1*d2,axis=1));add=np.log(np.sum(center*d1,axis=1))+np.log(np.sum(center*d2,axis=1))
    return {'center_posterior_entropy_mean':float(np.mean(-np.sum(post*np.log(post),axis=1))),
       'center_unary_entropy_mean':float(np.mean(-np.sum(center*np.log(center),axis=1))),
       'posterior_unary_tv_mean':float(np.mean(.5*np.sum(np.abs(post-center),axis=1))),
       'joint_minus_add_score_mean':float(np.mean(joint-add)),'joint_minus_add_score_abs_mean':float(np.mean(np.abs(joint-add))),
       'replace_second_with_first_abs_score_delta_mean':float(np.mean(np.abs(joint-np.log(np.sum(center*d1*d1,axis=1))))),
       'coupling':[float(x) for x in g],'near_saturation_count':int(np.sum(g>1.49))}

if __name__=='__main__':
    data_id,arm,seed=sys.argv[1],sys.argv[2],int(sys.argv[3])
    if arm not in ARMS or seed not in SEEDS:raise ValueError('outside fixed panel')
    start=time.monotonic();mlflow.set_tracking_uri(tracking_uri());a,m=load(data_id);prior=.25*np.log(a['counts'].astype(float));jax,jnp,_=_imports()

    if jax.default_backend()!='cpu':raise ValueError('protocol requires CPU backend')
    params=initial(arm,seed,len(prior));score,scores,loss,step,opt=functions(arm,prior);state=opt.init(params);count=sum(np.asarray(v).size for v in params.values())
    train=a['shuffled_windows'] if arm=='joint_shuffled' else a['train_windows'];exp=mlflow.set_experiment(EXP)
    matches=[r for r in MlflowClient().search_runs([exp.experiment_id],max_results=1000) if r.data.tags.get('arm')==arm and r.data.tags.get('seed')==str(seed)]
    if matches:raise ValueError('duplicate run; inspect before retry')
    source_hash=hashlib.sha256(''.join(m['source_sha256'].values()).encode()).hexdigest()
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='wikitext_joint_'+arm,
      energy_convention='mean vector dot + host center prior' if arm=='vector' else 'normalized 6-spin star or sum F2 + host center prior',
      variable_encoding='3D shared vector' if arm=='vector' else 'spins {-1,+1}',beta_temperature_convention='not applicable' if arm=='vector' else 'beta=1',
      sampler_schedule='exact 4/16/64-state factorization, Adam 3 epochs batch256',seed=seed,data_hash=source_hash,
      data_split='WikiText train and validation windows; all tests untouched',graph_summary=('shared 3D vector' if arm=='vector' else 'two independent 4-spin graphs, eight total edges, max degree2' if arm=='pair_add' else '6 spins, seven edges, max degree3 (joint)'),limitations=m['limitations'])
    details={'dataset_run_id':data_id,'dataset_hashes':m['hashes'],'base_dataset_run':m['base_dataset_run'],'arm':arm,'seed':seed,
      'parameter_count':count,'train_hash':sha_array(train),'offset_sha256':sha_array(prior),'epochs':3,'batch':256,'lr':.01,'clip':5.,
      'checkpoint_rule':'epoch3','metrics':{},'diagnostics':{},'checkpoint_hashes':{},'run_complete':False,
      'backend':jax.default_backend(),'devices':[str(d) for d in jax.devices()],
      'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','JAX_PLATFORMS')}}
    with mlflow.start_run(run_name=f'wikitext-joint-{arm}-seed{seed}') as run:
        mlflow.set_tags({'arm':arm,'seed':str(seed),'dataset_run_id':data_id,'test_status':'reserved not evaluated'})
        mlflow.log_params({'arm':arm,'seed':seed,'parameter_count':count,'epochs':3,'batch':256,'lr':.01,'clip':5.,'dataset_run_id':data_id})
        log_contract(mlflow,contract,details)
        for rel in ('scripts/train_wikitext_joint_context.py','scripts/build_wikitext2_joint_context_data.py','scripts/run_wikitext_joint_context.py','word_energy/wikitext_joint_context.py','word_energy/wikitext_joint_evaluation.py','word_energy/text8_factorial_models.py','word_energy/wikitext_matched_vectors.py','word_energy/text8_lexical_models.py','scripts/build_wikitext2_document_data.py','word_energy/ising/exact_training.py','notes/WIKITEXT2_JOINT_CONTEXT_PROTOCOL.md','tests/test_wikitext_joint_context.py','tests/test_wikitext_joint_evaluation.py'):
            mlflow.log_artifact(str(ROOT/rel),'source_snapshot')
        w=a['validation_windows'];n=a['validation_noise'];neg=np.broadcast_to(w[:,None,:],(len(w),n.shape[1],3)).copy();neg[:,:,0]=n
        if arm=='vector':
            zero={k:jnp.zeros_like(v) for k,v in params.items()};zero_score,_,_,_,_=functions(arm,prior)
            zp=np.asarray(zero_score(zero,w));zn=np.asarray(zero_score(zero,neg.reshape(-1,3))).reshape(n.shape)
        else:
            @jax.jit
            def zero_score(x):return score_windows(params,x,prior,arm,coupling_scale=0.)
            zp=np.asarray(zero_score(w));zn=np.asarray(zero_score(neg.reshape(-1,3))).reshape(n.shape)
        if not np.array_equal(zp,prior[w[:,0]]) or not np.array_equal(zn,prior[n]):raise ValueError('zero coupling not exact frequency prior')
        for key in ('noise_pairwise_auc','noise_mrr_at_21'):
            control=pair_metrics(zp,zn,a['counts'][w[:,0]])
            if not np.isclose(control[key],m['frequency_only'][key],atol=1e-14,rtol=0):raise ValueError('frequency control mismatch')
        details['zero_control']=control;tmp=ROOT/'.tmp';tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as d:
            tmp=Path(d);details['checkpoint_hashes']['initial']=snapshot(params,'initial',tmp);details['metrics']['initial']=validation(params,score,loss,a);log_metrics('initial',details['metrics']['initial'])
            for epoch in range(3):
                total=0.;n_seen=0
                for i in range(0,len(train),256):
                    ix=a['orders'][epoch,i:i+256];params,state,value=step(params,state,train[ix],a['train_noise'][epoch,ix]);value=float(value)
                    if not np.isfinite(value):raise FloatingPointError('nonfinite loss')
                    total+=len(ix)*value;n_seen+=len(ix)
                peak=peak_working_set()
                if peak>2*1024**3:raise MemoryError('2GiB working set check')
                stage=f'epoch{epoch+1}';details['checkpoint_hashes'][stage]=snapshot(params,stage,tmp);metrics=validation(params,score,loss,a)
                metrics['train_objective_mean']=total/n_seen;metrics['peak_working_set_mib']=peak/1024**2;details['metrics'][stage]=metrics;log_metrics(stage,metrics)
                if arm!='vector':
                    details['diagnostics'][stage]=diagnostics(params,a)
                    details['diagnostics'][stage]['readout']='joint star counterfactual for additive-trained parameters' if arm=='pair_add' else 'joint star model'
                print(json.dumps({'run_id':run.info.run_id,'arm':arm,'seed':seed,'stage':stage,'auc':metrics['noise_pairwise_auc'],'macro':metrics['document_macro_auc'],'mrr':metrics['noise_mrr_at_21']}),flush=True)
            details['elapsed_seconds']=time.monotonic()-start;details['run_complete']=True;log_contract(mlflow,contract,details);mlflow.log_metric('resource.elapsed_seconds',details['elapsed_seconds'])
