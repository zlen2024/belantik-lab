"""Audit learned 12-spin star distributions in THRML against exact law."""
from pathlib import Path
from itertools import combinations
import json,sys,tempfile,time,traceback
import numpy as np
from scipy.special import expit,logsumexp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.build_wikitext2_joint_context_data import load
from word_energy.ising.exact_training import _imports
from word_energy.ising.fft_diagnostics import split_rank_rhat_ess_fft
from word_energy.wikitext_four_spin import score_windows
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file
DATA='f1b7db5231ed4a5788b566e388f8e731';EXP='word-energy-thrml-four-spin-parity';MODEL_EXP='word-energy-wikitext2-four-spin-context'
STATES=np.array([[2*((i>>j)&1)-1 for j in range(12)] for i in range(4096)],dtype=np.float64)
SINGLE=STATES[:16,:4];PAIRS=tuple(combinations(range(4),2))
EDGES=[(base+i,base+j) for base in (0,4,8) for i,j in PAIRS]+[(i,base+i) for base in (4,8) for i in range(4)]
BLOCKS=[[i,4+(i+1)%4,8+(i+1)%4] for i in range(4)]

def graph_check():
 if len(EDGES)!=26 or len({v for block in BLOCKS for v in block})!=12:raise ValueError('bad graph coverage')
 for block in BLOCKS:
  if any(i in block and j in block for i,j in EDGES):raise ValueError('invalid Gibbs coloring')

def make_runner(bias,weights):
 jax,jnp,_=_imports()
 from thrml import Block,SamplingSchedule,SpinNode,sample_states
 from thrml.models import IsingEBM,IsingSamplingProgram
 graph_check();nodes=[SpinNode() for _ in range(12)];edges=[(nodes[i],nodes[j]) for i,j in EDGES]
 model=IsingEBM(nodes,edges,jnp.asarray(bias),jnp.asarray(weights),jnp.array(1.));blocks=[Block([nodes[i] for i in block]) for block in BLOCKS]
 program=IsingSamplingProgram(model,blocks,[]);schedule=SamplingSchedule(n_warmup=500,n_samples=8000,steps_per_sample=1)
 starts=jnp.asarray([[False]*12,[True]*12,[bool(i%2) for i in range(12)],[not bool(i%2) for i in range(12)]],dtype=jnp.bool_)
 init=[starts[:,block] for block in BLOCKS]
 runner=jax.jit(jax.vmap(lambda ini,key:sample_states(key,program,schedule,ini,[],[Block(nodes)])))
 def sample(seed):return 2*np.asarray(runner(init,jax.random.split(jax.random.key(seed),4))[0],dtype=np.int8)-1
 return sample

def diag(values,reference):
 rh,ess,mcse=split_rank_rhat_ess_fft(values);emp=float(np.mean(values));finite=bool(np.isfinite(rh) and np.isfinite(mcse))
 return {'exact':float(reference),'sample':emp,'rhat':float(rh) if finite else None,'ess':float(ess),'mcse':float(mcse) if finite else None,
  'pass':bool(finite and rh<1.05 and ess>=400 and mcse<=.01 and abs(emp-reference)<=max(.02,3*mcse))}

def audit():
 jax,_,_=_imports()
 if jax.default_backend()!='cpu':raise ValueError('protocol requires CPU')
 c=MlflowClient();a,m=load(DATA);exp=c.get_experiment_by_name(MODEL_EXP)
 rows=[r for r in c.search_runs([exp.experiment_id],max_results=1000) if r.data.tags.get('arm')=='joint4' and r.data.tags.get('seed')=='17']
 if len(rows)!=1 or rows[0].info.status!='FINISHED':raise ValueError('missing trained model')
 run=rows[0];root=artifact_dir(c,run.info.run_id);cfg=json.loads((root/'config.json').read_text(encoding='utf8'))
 if not cfg['run_complete'] or cfg['dataset_hashes']!=m['hashes']:raise ValueError('dataset mismatch')
 path=ROOT/'word_energy/wikitext_four_spin.py'
 if sha256_file(path)!=sha256_file(root/'source_snapshot'/path.name):raise ValueError('model source changed')
 with np.load(root/'checkpoints/epoch3.npz',allow_pickle=False) as z:p={k:z[k] for k in z.files}
 if {k:sha_array(v) for k,v in p.items()}!=cfg['checkpoint_hashes']['epoch3']:raise ValueError('checkpoint mismatch')
 vocabulary={w:i for i,w in enumerate(m['vocabulary'])};results={};saved={}
 for fixture,words in [('river',('bank','river','water')),('finance',('bank','money','financial'))]:
  ids=[vocabulary[w] for w in words];b=2.5*np.tanh(p['biases'][ids]).reshape(-1);internal=2*np.tanh(p['internal'][ids]);g=1.5*expit(p['interaction'])/2
  weights=np.r_[internal.reshape(-1),g,g];logw=STATES@b
  for (i,j),weight in zip(EDGES,weights):logw+=weight*STATES[:,i]*STATES[:,j]
  prob=np.exp(logw-logsumexp(logw));unary=[]
  for q in range(3):
   u=SINGLE@b[4*q:4*q+4]
   for (i,j),v in zip(PAIRS,internal[q]):u+=v*SINGLE[:,i]*SINGLE[:,j]
   unary.append(float(logsumexp(u)))
  exact_score=float(logsumexp(logw)-sum(unary));code_score=float(np.asarray(score_windows(p,np.array([ids],np.int32),np.zeros(len(vocabulary)),'joint4'))[0])
  if not np.isclose(exact_score,code_score,atol=1e-11,rtol=0):raise ValueError('exact partition mismatch')
  rng=np.random.default_rng(24700 if fixture=='river' else 24701);iid=rng.multinomial(32000,prob,size=128)/32000;iid_tv=.5*np.abs(iid-prob).sum(axis=1)
  sampler=make_runner(b,weights);runs=[]
  for seed in (17,29,43):
   x=sampler(seed);saved[f'{fixture}_{seed}']=x
   idx=np.sum((x>0)*(1<<np.arange(12)),axis=2);emp=np.bincount(idx.ravel(),minlength=4096)/idx.size
   observables={}
   for i in range(12):observables[f'spin{i}_plus']=diag((x[:,:,i]>0).astype(float),float(prob[STATES[:,i]>0].sum()))
   for i,j in EDGES[-8:]:observables[f'edge{i}_{j}']=diag((x[:,:,i]*x[:,:,j]).astype(float),float(prob@(STATES[:,i]*STATES[:,j])))
   runs.append({'seed':seed,'state_probability':emp.tolist(),'target_marginal':np.bincount((idx%16).ravel(),minlength=16).tolist(),
      'total_variation':float(.5*np.abs(emp-prob).sum()),'observables':observables,'all_observables_pass':all(v['pass'] for v in observables.values())})
   runs[-1]['target_marginal']=[v/idx.size for v in runs[-1]['target_marginal']]
  results[fixture]={'words':words,'ids':ids,'bias':b.tolist(),'weights':weights.tolist(),'exact_state_probability':prob.tolist(),
    'exact_normalized_score_without_host_prior':exact_score,'production_score_without_host_prior':code_score,
    'iid_tv_mean':float(iid_tv.mean()),'iid_tv_95':float(np.percentile(iid_tv,95)),'runs':runs}
 return {'backend':jax.default_backend(),'devices':[str(d) for d in jax.devices()],'run_complete':True,'dataset_run_id':DATA,'source_model_run':run.info.run_id,'checkpoint_hashes':cfg['checkpoint_hashes']['epoch3'],'graph_edges':EDGES,'color_blocks':BLOCKS,'results':results,
  'limitations':'THRML CPU simulator 2 triples, no physical hardware; samples verify state/moment law, not free-energy estimation or host-prior hardware mapping'},saved

if __name__=='__main__':
 start=time.monotonic();mlflow.set_tracking_uri(tracking_uri());mlflow.set_experiment(EXP)
 contract=RunContract(execution_tier='thrml_simulator',device_class='cpu_gpu_simulator',model_family='wikitext_four_spin_joint_parity',energy_convention='E=-sum b*s-sum J*s*s beta1',variable_encoding='THRML bool True=+1',beta_temperature_convention='beta=1',sampler_schedule='4-color block Gibbs, warmup500, draws8000, step1',seed=17,data_hash=DATA,data_split='trained development checkpoint, two illustrative vocabulary triples; tests untouched',graph_summary='12 spins, 26 edges, degree5, four colors',limitations='THRML CPU simulator, two triples; sampled moments not free energy or chip execution',chain_count=4,burn_in=500,sample_count=8000)
 with mlflow.start_run(run_name='four-spin-learned-joint-parity') as run:
  log_contract(mlflow,contract,{'run_complete':False,'dataset_run_id':DATA})
  for p in (Path(__file__),ROOT/'notes/THRML_FOUR_SPIN_PARITY_PROTOCOL.md',ROOT/'word_energy/ising/fft_diagnostics.py',ROOT/'word_energy/wikitext_four_spin.py',ROOT/'tests/test_thrml_four_spin_joint.py'):mlflow.log_artifact(str(p),'source_snapshot')
  tmp=ROOT/'.tmp';tmp.mkdir(exist_ok=True)
  try:
   out,samples=audit();out['elapsed_seconds']=time.monotonic()-start;log_contract(mlflow,contract,out)
   with tempfile.TemporaryDirectory(dir=tmp) as d:
    p=Path(d)/'spin_samples.npz';np.savez_compressed(p,**samples);mlflow.log_artifact(str(p),'samples')
   print(json.dumps({'run_id':run.info.run_id,'summary':{k:{'exact_score':v['exact_normalized_score_without_host_prior'],'iid_tv_mean':v['iid_tv_mean'],'runs':[{'seed':r['seed'],'tv':r['total_variation'],'all_pass':r['all_observables_pass']} for r in v['runs']]} for k,v in out['results'].items()}},indent=2))
  except BaseException:
   with tempfile.TemporaryDirectory(dir=tmp) as d:
    p=Path(d)/'failure.txt';p.write_text(traceback.format_exc(),encoding='utf8');mlflow.log_artifact(str(p),'interruption')
   raise
