"""Repeated WiC-development subset for four-spin and matched vector models."""
from pathlib import Path
import json,sys,time
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.build_wikitext2_joint_context_data import load
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.evaluate_wic_context_probe import bootstrap
from scripts.evaluate_wikitext_joint_wic import paired_bootstrap
from word_energy.wikitext_four_spin_wic import unary,posterior
from word_energy.wic_context_probe import negative_js,cosine,auc
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file
DATA='f1b7db5231ed4a5788b566e388f8e731';REFERENCE='fa6a014eefc549338b7735a9e52206ed';TWO='48704618227d4d98b98056908c3b5591';EXP='word-energy-wikitext2-four-spin-context';SEEDS=(17,29,43)

def audit():
 c=MlflowClient();_,meta=load(DATA);root=artifact_dir(c,REFERENCE);ref=json.loads((root/'config.json').read_text(encoding='utf8'))
 if meta['base_hashes']!=ref['dataset_hashes']:raise ValueError('vocabulary/reference base hashes differ')
 for rel in ('word_energy/wic_context_probe.py','scripts/evaluate_wic_context_probe.py'):
  p=ROOT/rel
  if sha256_file(p)!=sha256_file(root/'source_snapshot'/p.name):raise ValueError('historical helper mismatch')
 two_root=artifact_dir(c,TWO);two=json.loads((two_root/'config.json').read_text(encoding='utf8'))
 if two['dataset_run_id']!=DATA or two['wic_reference_run']!=REFERENCE:raise ValueError('two-spin probe mismatch')
 selected=ref['selected'];labels=np.asarray(ref['labels'],np.int32);groups=[r['word'] for r in selected]
 if len(selected)!=167 or len(labels)!=167 or labels.sum()!=82:raise ValueError('reference subset mismatch')
 exp=c.get_experiment_by_name(EXP);runs=c.search_runs([exp.experiment_id],max_results=1000);predictions={};sources={}
 for arm,stages in [('joint4',('initial','epoch3')),('pair_add4',('epoch3',)),('joint4_shuffled',('epoch3',)),('vector10',('initial','epoch3'))]:
  for stage in stages:
   scores=[];entropy=[];context_scores=[]
   for seed in SEEDS:
    matches=[r for r in runs if r.data.tags.get('arm')==arm and r.data.tags.get('seed')==str(seed)]
    if len(matches)!=1 or matches[0].info.status!='FINISHED':raise ValueError('missing/duplicate run')
    run=matches[0];modelroot=artifact_dir(c,run.info.run_id);cfg=json.loads((modelroot/'config.json').read_text(encoding='utf8'))
    if not cfg['run_complete'] or cfg['dataset_run_id']!=DATA or cfg['dataset_hashes']!=meta['hashes']:raise ValueError('training data mismatch')
    with np.load(modelroot/'checkpoints'/f'{stage}.npz',allow_pickle=False) as z:p={k:z[k] for k in z.files}
    if {k:sha_array(v) for k,v in p.items()}!=cfg['checkpoint_hashes'][stage]:raise ValueError('checkpoint mismatch')
    sources[f'{arm}:{stage}:{seed}']=run.info.run_id;values=[];ent=[];ctx_values=[]
    if arm!='vector10':q=unary(p);g=1.5*expit(p['interaction'])
    for r in selected:
     t=r['target'];c1=r['context1'];c2=r['context2']
     if arm=='vector10':
      v=p['shared'];b1=np.mean(v[c1],axis=0);b2=np.mean(v[c2],axis=0);values.append(cosine(v[t]+b1,v[t]+b2));ctx_values.append(cosine(b1,b2))
     else:
      x=posterior(q,g,t,c1);y=posterior(q,g,t,c2);values.append(negative_js(x,y));ent.extend([-np.sum(x*np.log(x)),-np.sum(y*np.log(y))])
    scores.append(values)
    if ent:entropy.append(float(np.mean(ent)))
    if ctx_values:context_scores.append(ctx_values)
   predictions[f'{arm}:{stage}']={'scores':scores,'posterior_entropy':entropy}
   if context_scores:predictions[f'{arm}:{stage}:context_only']={'scores':context_scores}
 for key in ('joint:epoch3','pair_add:epoch3','vector:epoch3','joint:initial','previous:bag_jaccard'):
  predictions['two_spin:'+key]=two['predictions'][key]
 summaries={}
 for key,v in predictions.items():
  scores=[np.asarray(x,dtype=float) for x in v['scores']]
  if any(not np.isfinite(x).all() or len(x)!=len(labels) for x in scores):raise ValueError('invalid scores')
  values=[auc(labels,x) for x in scores];summaries[key]={'auc_per_seed':values,'auc_mean':float(np.mean(values)),'bootstrap':bootstrap(labels,groups,scores)}
  if v.get('posterior_entropy'):summaries[key]['posterior_entropy_mean']=v['posterior_entropy']
 effects={};joint=predictions['joint4:epoch3']['scores']
 for k in ('pair_add4:epoch3','joint4:initial','joint4_shuffled:epoch3','vector10:epoch3','two_spin:joint:epoch3','two_spin:previous:bag_jaccard'):
  effects['joint4_minus_'+k]=paired_bootstrap(labels,groups,joint,predictions[k]['scores'])
 return {'dataset_run_id':DATA,'dataset_hashes':meta['hashes'],'wic_reference_run':REFERENCE,'two_spin_wic_run':TWO,'wic_sha256':ref['wic_sha256'],'source_runs':sources,
  'selected_count':len(selected),'total_examples':ref['total_examples'],'positive':int(labels.sum()),'negative':int(len(labels)-labels.sum()),'target_count':len(set(groups)),
  'predictions':predictions,'summaries':summaries,'paired_effects':effects,
  'limitations':'repeated inspected 167-example WiC development; 4-spin readout untrained on WiC, 1-4 context cardinality, no unbiased benchmark or physical hardware claim'}

if __name__=='__main__':
 start=time.monotonic();mlflow.set_tracking_uri(tracking_uri());out=audit();out['elapsed_seconds']=time.monotonic()-start;mlflow.set_experiment(EXP)
 contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='four_spin_repeated_wic_probe',energy_convention='normalized 16-state star posterior or 10D vector cosine',variable_encoding='four-spin Ising or 10D vector',beta_temperature_convention='Ising beta1',sampler_schedule='exact posterior, no WiC training',seed=22150,data_hash=out['wic_sha256'],data_split='previously inspected WiC dev subset only; tests untouched',graph_summary='4 spins per word, up to four context spokes',limitations=out['limitations'])
 with mlflow.start_run(run_name='four-spin-repeated-wic-development') as run:
  log_contract(mlflow,contract,out)
  for k,v in out['summaries'].items():mlflow.log_metric(k.replace(':','.')+'.auc_mean',v['auc_mean'])
  for p in (Path(__file__),ROOT/'word_energy/wikitext_four_spin_wic.py',ROOT/'word_energy/wic_context_probe.py',ROOT/'scripts/evaluate_wikitext_joint_wic.py',ROOT/'scripts/evaluate_wic_context_probe.py',ROOT/'tests/test_wikitext_four_spin_wic.py',ROOT/'notes/WIKITEXT2_FOUR_SPIN_PROTOCOL.md'):mlflow.log_artifact(str(p),'source_snapshot')
  print(json.dumps({'run_id':run.info.run_id,'selected':out['selected_count'],'summaries':out['summaries'],'paired_effects':out['paired_effects']},indent=2))
