"""Audit four-spin endpoints against original two-spin and matched vector."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.build_wikitext2_joint_context_data import load
from scripts.build_wikitext2_document_data import doc_macro
from word_energy.wikitext_four_spin import ARMS,SEEDS,functions as four_functions
from word_energy.wikitext_joint_context import functions as two_functions
from word_energy.wikitext_joint_evaluation import canonical_scores
from word_energy.text8_lexical_models import pair_metrics
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file
DATA='f1b7db5231ed4a5788b566e388f8e731';EXP='word-energy-wikitext2-four-spin-context';OLD='word-energy-wikitext2-joint-context'
MAPPING={'joint4':'joint','pair_add4':'pair_add','vector10':'vector','joint4_shuffled':'joint_shuffled'}

def audit():
 c=MlflowClient();a,m=load(DATA);newexp=c.get_experiment_by_name(EXP);oldexp=c.get_experiment_by_name(OLD)
 new=[r for r in c.search_runs([newexp.experiment_id],max_results=1000) if r.data.tags.get('arm') in ARMS]
 old=[r for r in c.search_runs([oldexp.experiment_id],max_results=1000) if r.data.tags.get('arm') in MAPPING.values()]
 if len(new)!=12 or len(old)!=12:raise ValueError('expected 12+12 trained runs')
 w=a['validation_windows'];noise=a['validation_noise'];counts=a['counts'];docs=a['validation_document'];docids=np.unique(docs);prior=.25*np.log(counts.astype(float))
 scorers={arm:four_functions(arm,prior)[0] for arm in ARMS};scorers.update({arm:two_functions(arm,prior)[0] for arm in MAPPING.values()})
 results={};docscores={};inits={};sources=None
 for group,records in [('four',new),('two',old)]:
  for run in records:
   arm=run.data.tags['arm'];seed=int(run.data.tags['seed']);key=f'{arm}:{seed}'
   if key in results or run.info.status!='FINISHED':raise ValueError('duplicate or unfinished')
   root=artifact_dir(c,run.info.run_id);cfg=json.loads((root/'config.json').read_text(encoding='utf8'))
   if not cfg['run_complete'] or cfg['dataset_run_id']!=DATA or cfg['dataset_hashes']!=m['hashes'] or cfg['offset_sha256']!=sha_array(prior):raise ValueError('data/config mismatch')
   shuffled=arm in ('joint4_shuffled','joint_shuffled');expected=a['shuffled_windows'] if shuffled else a['train_windows']
   if cfg['train_hash']!=sha_array(expected) or (cfg['epochs'],cfg['batch'],cfg['lr'],cfg['clip'])!=(3,256,.01,5.):raise ValueError('training protocol differs')
   if group=='four':
    count=20000 if arm=='vector10' else 20004
    if cfg['parameter_count']!=count or cfg['backend']!='cpu':raise ValueError('model size/backend')
    source={p.name:sha256_file(p) for p in (root/'source_snapshot').iterdir() if p.is_file()}
    if sources is not None and source!=sources:raise ValueError('source snapshots differ')
    sources=source
    for rel in ('word_energy/wikitext_four_spin.py','word_energy/wikitext_four_spin_diagnostics.py','word_energy/wikitext_joint_evaluation.py','word_energy/text8_lexical_models.py','word_energy/ising/exact_training.py'):
     p=ROOT/rel
     if sha256_file(p)!=source[p.name]:raise ValueError('scorer changed')
   loaded={}
   for stage in ('initial','epoch1','epoch2','epoch3'):
    with np.load(root/'checkpoints'/f'{stage}.npz',allow_pickle=False) as z:p={k:z[k] for k in z.files}
    if {k:sha_array(v) for k,v in p.items()}!=cfg['checkpoint_hashes'][stage]:raise ValueError('checkpoint hash mismatch')
    if stage in ('initial','epoch3'):loaded[stage]=p
   inits[key]=loaded['initial'];pos,neg=canonical_scores(loaded['epoch3'],scorers[arm],w,noise);metrics=pair_metrics(pos,neg,counts[w[:,0]]);metrics['document_macro_auc']=doc_macro(pos,neg,docs)
   for k in ('noise_pairwise_auc','noise_mrr_at_21','document_macro_auc'):
    if not np.isclose(metrics[k],cfg['metrics']['epoch3'][k],atol=1e-14,rtol=0):raise ValueError('metric mismatch')
   wins=((pos[:,None]>neg)+.5*(pos[:,None]==neg)).mean(axis=1);docscores[key]=np.array([wins[docs==d].mean() for d in docids])
   results[key]={'run_id':run.info.run_id,'metrics':metrics,'initial_metrics':cfg['metrics']['initial'],'diagnostics':cfg['diagnostics'].get('epoch3',{}),'initial_diagnostics':cfg['diagnostics'].get('initial',{}),'parameter_count':cfg['parameter_count'],'elapsed_seconds':cfg['elapsed_seconds'],'peak_mib':cfg['metrics']['epoch3']['peak_working_set_mib'],'reused':group=='two'}
   if group=='four':
    paired=[r for r in old if r.data.tags.get('arm')==MAPPING[arm] and r.data.tags.get('seed')==str(seed)]
    if len(paired)!=1 or cfg['paired_two_spin_run_id']!=paired[0].info.run_id:raise ValueError('incorrect two-spin reference')
 for seed in SEEDS:
  x=inits[f'joint4:{seed}']
  for arm in ('pair_add4','joint4_shuffled'):
   y=inits[f'{arm}:{seed}']
   if set(x)!=set(y) or any(not np.array_equal(v,y[k]) for k,v in x.items()):raise ValueError('four-spin paired init mismatch')
 rng=np.random.default_rng(23140);indices=rng.integers(0,len(docids),size=(1000,len(docids)));effects={}
 for left,right in [('joint4','joint'),('joint4','pair_add4'),('joint4','vector10'),('joint4','joint4_shuffled'),('pair_add4','pair_add'),('vector10','vector')]:
  micro=[results[f'{left}:{s}']['metrics']['noise_pairwise_auc']-results[f'{right}:{s}']['metrics']['noise_pairwise_auc'] for s in SEEDS]
  mrr=[results[f'{left}:{s}']['metrics']['noise_mrr_at_21']-results[f'{right}:{s}']['metrics']['noise_mrr_at_21'] for s in SEEDS]
  d=np.mean([docscores[f'{left}:{s}']-docscores[f'{right}:{s}'] for s in SEEDS],axis=0)
  effects[f'{left}_minus_{right}']={'micro_per_seed':micro,'mrr_per_seed':mrr,'macro_difference':float(d.mean()),'article_bootstrap_95_conditional':[float(x) for x in np.percentile(d[indices].mean(axis=1),[2.5,97.5])]}
 return {'dataset_run_id':DATA,'dataset_hashes':m['hashes'],'source_sha256':m['source_sha256'],'source_snapshots_four':sources,'results':results,'effects':effects,'frequency_only':m['frequency_only'],'n_validation_articles':len(docids),
  'limits':'posthoc development; 4 vs 2 changes capacity and topology; vector count match only; 3 seeds share data; no test/sense/hardware claim'}

if __name__=='__main__':
 start=time.monotonic();mlflow.set_tracking_uri(tracking_uri());out=audit();out['elapsed_seconds']=time.monotonic()-start;mlflow.set_experiment(EXP)
 contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='wikitext_four_spin_summary',energy_convention='normalized 4/2-spin F3/F2 or shared vector + center prior',variable_encoding='spins {-1,+1} or real vectors',beta_temperature_convention='Ising beta1',sampler_schedule='exact canonical checkpoint audit',seed=23140,data_hash=DATA,data_split='WikiText development train/validation, tests untouched',graph_summary='12-spin or 6-spin star, or shared vector',limitations=out['limits'])
 with mlflow.start_run(run_name='four-spin-context-summary') as run:
  log_contract(mlflow,contract,out)
  for k,v in out['effects'].items():mlflow.log_metric(k+'.macro_auc_diff',v['macro_difference'])
  for p in (Path(__file__),ROOT/'notes/WIKITEXT2_FOUR_SPIN_PROTOCOL.md'):mlflow.log_artifact(str(p),'source_snapshot')
  print(json.dumps({'run_id':run.info.run_id,'frequency_only':out['frequency_only'],'effects':out['effects'],'results':out['results']},indent=2))
