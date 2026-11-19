"""One 4-spin Ising or matched 10D vector run on frozen joint data."""
from pathlib import Path
import hashlib,json,os,sys,tempfile,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.build_wikitext2_joint_context_data import load
from scripts.train_wikitext_joint_context import validation
from scripts.train_text8_lexical_development import snapshot,log_metrics
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.run_ising_corrected_joint import peak_working_set
from word_energy.ising.exact_training import _imports
from word_energy.wikitext_four_spin import ARMS,SEEDS,initial,functions,score_windows
from word_energy.wikitext_four_spin_diagnostics import diagnostics
from word_energy.text8_lexical_models import pair_metrics
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file
DATA='f1b7db5231ed4a5788b566e388f8e731';EXP='word-energy-wikitext2-four-spin-context'

if __name__=='__main__':
 arm,seed=sys.argv[1],int(sys.argv[2]);
 if arm not in ARMS or seed not in SEEDS:raise ValueError('outside frozen panel')
 start=time.monotonic();mlflow.set_tracking_uri(tracking_uri());a,m=load(DATA);prior=.25*np.log(a['counts'].astype(float));jax,jnp,_=_imports()
 if jax.default_backend()!='cpu':raise ValueError('CPU protocol')
 params=initial(arm,seed,len(prior));score,scores,loss,step,opt=functions(arm,prior);state=opt.init(params);count=sum(np.asarray(v).size for v in params.values())
 train=a['shuffled_windows'] if arm=='joint4_shuffled' else a['train_windows'];client=MlflowClient();exp=mlflow.set_experiment(EXP)
 if any(r.data.tags.get('arm')==arm and r.data.tags.get('seed')==str(seed) for r in client.search_runs([exp.experiment_id],max_results=1000)):raise ValueError('duplicate attempt; inspect')
 old_exp=client.get_experiment_by_name('word-energy-wikitext2-joint-context');old_arm={'joint4':'joint','pair_add4':'pair_add','vector10':'vector','joint4_shuffled':'joint_shuffled'}[arm]
 rows=[r for r in client.search_runs([old_exp.experiment_id],max_results=1000) if r.data.tags.get('arm')==old_arm and r.data.tags.get('seed')==str(seed)]
 if len(rows)!=1 or rows[0].info.status!='FINISHED':raise ValueError('missing reference run')
 old=rows[0];oldroot=artifact_dir(client,old.info.run_id);oldcfg=json.loads((oldroot/'config.json').read_text(encoding='utf8'))
 if not oldcfg['run_complete'] or oldcfg['dataset_run_id']!=DATA or oldcfg['dataset_hashes']!=m['hashes'] or oldcfg['train_hash']!=sha_array(train) or oldcfg['offset_sha256']!=sha_array(prior):raise ValueError('reference dataset mismatch')
 if (oldcfg['epochs'],oldcfg['batch'],oldcfg['lr'],oldcfg['clip'])!=(3,256,.01,5.):raise ValueError('reference budget mismatch')
 with np.load(oldroot/'checkpoints/epoch3.npz',allow_pickle=False) as z:oldparams={k:z[k] for k in z.files}
 if {k:sha_array(v) for k,v in oldparams.items()}!=oldcfg['checkpoint_hashes']['epoch3']:raise ValueError('reference checkpoint mismatch')
 details={'dataset_run_id':DATA,'dataset_hashes':m['hashes'],'arm':arm,'seed':seed,'paired_two_spin_run_id':old.info.run_id,'parameter_count':count,
  'train_hash':sha_array(train),'offset_sha256':sha_array(prior),'epochs':3,'batch':256,'lr':.01,'clip':5.,'checkpoint_rule':'epoch3',
  'backend':jax.default_backend(),'devices':[str(d) for d in jax.devices()],'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','JAX_PLATFORMS')},
  'metrics':{},'diagnostics':{},'checkpoint_hashes':{},'run_complete':False}
 source_hash=hashlib.sha256(''.join(m['source_sha256'].values()).encode()).hexdigest()
 vector=arm=='vector10';add=arm=='pair_add4'
 graph='shared 10D vector' if vector else ('two disjoint 8-spin pair graphs, 32 total edges, max degree4' if add else '12 spins, 26 edges, max degree5')
 contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='wikitext_four_spin_'+arm,
  energy_convention='10D dot + host center prior' if vector else 'normalized 12-spin star or independent 8-spin pairs + host center prior',
  variable_encoding='shared 10D vector' if vector else 'spins {-1,+1}',beta_temperature_convention='not applicable' if vector else 'beta=1',
  sampler_schedule='exact 16-state factors, Adam 3 epochs batch256',seed=seed,data_hash=source_hash,data_split='WikiText development train/validation; tests untouched',graph_summary=graph,
  limitations='posthoc development, 4 vs 2 changes topology and capacity; no semantic or physical-chip claim')
 with mlflow.start_run(run_name=f'wikitext-four-spin-{arm}-seed{seed}') as run:
  mlflow.set_tags({'arm':arm,'seed':str(seed),'dataset_run_id':DATA,'paired_two_spin_run_id':old.info.run_id,'test_status':'reserved not evaluated'})
  mlflow.log_params({'arm':arm,'seed':seed,'parameter_count':count,'epochs':3,'batch':256,'lr':.01,'clip':5.,'dataset_run_id':DATA})
  log_contract(mlflow,contract,details)
  for rel in ('scripts/train_wikitext_four_spin.py','scripts/run_wikitext_four_spin.py','word_energy/wikitext_four_spin.py','word_energy/wikitext_four_spin_diagnostics.py','scripts/train_wikitext_joint_context.py','scripts/build_wikitext2_joint_context_data.py','word_energy/wikitext_joint_evaluation.py','word_energy/text8_lexical_models.py','word_energy/ising/exact_training.py','notes/WIKITEXT2_FOUR_SPIN_PROTOCOL.md','tests/test_wikitext_four_spin.py','tests/test_wikitext_four_spin_diagnostics.py'):
   mlflow.log_artifact(str(ROOT/rel),'source_snapshot')
  w=a['validation_windows'];n=a['validation_noise'];neg=np.broadcast_to(w[:,None,:],(len(w),n.shape[1],3)).copy();neg[:,:,0]=n;neg=neg.reshape((-1,3))
  if vector:
   z={k:jnp.zeros_like(v) for k,v in params.items()};zero_score=lambda x:score(z,x)
  else:
   @jax.jit
   def zero_score(x):return score_windows(params,x,prior,arm,coupling_scale=0.)
  def chunks(x):return np.concatenate([np.asarray(zero_score(x[i:i+4096])) for i in range(0,len(x),4096)])
  zp=chunks(w);zn=chunks(neg).reshape(n.shape)
  if not np.array_equal(zp,prior[w[:,0]]) or not np.array_equal(zn,prior[n]):raise ValueError('zero coupling score mismatch')
  ctrl=pair_metrics(zp,zn,a['counts'][w[:,0]])
  for key in ('noise_pairwise_auc','noise_mrr_at_21'):
   if not np.isclose(ctrl[key],m['frequency_only'][key],atol=1e-14,rtol=0):raise ValueError('zero control mismatch')
  details['zero_control']=ctrl;tmp=ROOT/'.tmp';tmp.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=tmp) as d:
   t=Path(d);details['checkpoint_hashes']['initial']=snapshot(params,'initial',t);details['metrics']['initial']=validation(params,score,loss,a);log_metrics('initial',details['metrics']['initial'])
   if not vector:
    details['diagnostics']['initial']=diagnostics(params,a,m['vocabulary'])
    details['diagnostics']['initial']['readout']='joint counterfactual for additive parameters' if add else 'joint model'
   for epoch in range(3):
    total=0.;seen=0
    for i in range(0,len(train),256):
     ix=a['orders'][epoch,i:i+256];params,state,value=step(params,state,train[ix],a['train_noise'][epoch,ix]);value=float(value)
     if not np.isfinite(value):raise FloatingPointError('nonfinite loss')
     total+=len(ix)*value;seen+=len(ix)
    peak=peak_working_set()
    if peak>2*1024**3:raise MemoryError('2 GiB peak working set exceeded')
    stage=f'epoch{epoch+1}';details['checkpoint_hashes'][stage]=snapshot(params,stage,t);out=validation(params,score,loss,a)
    out['train_objective_mean']=total/seen;out['peak_working_set_mib']=peak/1024**2;details['metrics'][stage]=out;log_metrics(stage,out)
    if not vector:
     details['diagnostics'][stage]=diagnostics(params,a,m['vocabulary'])
     details['diagnostics'][stage]['readout']='joint counterfactual for additive parameters' if add else 'joint model'
    print(json.dumps({'run_id':run.info.run_id,'arm':arm,'seed':seed,'stage':stage,'auc':out['noise_pairwise_auc'],'macro':out['document_macro_auc'],'mrr':out['noise_mrr_at_21'],'peak_mib':out['peak_working_set_mib']}),flush=True)
   details['elapsed_seconds']=time.monotonic()-start;details['run_complete']=True;log_contract(mlflow,contract,details);mlflow.log_metric('resource.elapsed_seconds',details['elapsed_seconds'])
