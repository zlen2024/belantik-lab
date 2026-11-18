"""Audit matched 3D vectors against paired Ising checkpoints."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.build_wikitext2_document_data import load,doc_macro
from word_energy.wikitext_matched_vectors import functions as vector_fn
from word_energy.text8_factorial_models import functions as ising_fn
from word_energy.text8_canonical_eval import canonical_candidate_scores
from word_energy.text8_lexical_models import pair_metrics
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file
DATA='b37da8076fd54f4da389082278e9733a';SEEDS=(17,29,43);PAIRS=(('tied-frequency','shared3'),('role-frequency','directed3'))

def audit():
 c=MlflowClient();a,m=load(DATA);prior=.25*np.log(a['counts'].astype(float));pairs=a['validation_pairs'];noise=a['validation_noise'];docs=a['validation_document'];docids=np.unique(docs)
 results={};doc_values={};seed_init={}
 for ia,va in PAIRS:
  for seed in SEEDS:
   for kind,arm,experiment in [('ising',ia,'word-energy-wikitext2-doc-development'),('vector',va,'word-energy-wikitext2-matched-vectors')]:
    exp=c.get_experiment_by_name(experiment);rows=[r for r in c.search_runs([exp.experiment_id],max_results=1000) if r.data.tags.get('arm')==arm and r.data.tags.get('seed')==str(seed) and (kind=='vector' or r.data.tags.get('mode')=='original')]
    if len(rows)!=1 or rows[0].info.status!='FINISHED':raise ValueError('missing/duplicate run')
    r=rows[0];root=artifact_dir(c,r.info.run_id);cfg=json.loads((root/'config.json').read_text(encoding='utf8'))
    if not cfg['run_complete'] or cfg['dataset_run_id']!=DATA or cfg['dataset_hashes']!=m['hashes'] or cfg['offset_sha256']!=sha_array(prior):raise ValueError('config mismatch')
    for stage in ('initial','epoch1','epoch2','epoch3'):
     with np.load(root/'checkpoints'/f'{stage}.npz',allow_pickle=False) as z:p={k:z[k] for k in z.files}
     if {k:sha_array(v) for k,v in p.items()}!=cfg['checkpoint_hashes'][stage]:raise ValueError('checkpoint mismatch')
     if stage=='epoch3':final=p
    score=ising_fn(arm,prior)[0] if kind=='ising' else vector_fn(prior,arm=='shared3')[0]
    pos,neg=canonical_candidate_scores(final,score,pairs,noise,symmetric=False);metric=pair_metrics(pos,neg,a['counts'][pairs[:,0]]);metric['document_macro_auc']=doc_macro(pos,neg,docs)
    for key in ('noise_pairwise_auc','noise_mrr_at_21','document_macro_auc'):
     if not np.isclose(metric[key],cfg['metrics']['epoch3'][key],atol=1e-14,rtol=0):raise ValueError('metric mismatch')
    win=((pos[:,None]>neg)+.5*(pos[:,None]==neg)).mean(axis=1);key=f'{arm}:{seed}';doc_values[key]=np.array([win[docs==i].mean() for i in docids])
    results[key]={'run_id':r.info.run_id,'parameter_count':cfg['parameter_count'],'metrics':metric,'elapsed_seconds':cfg['elapsed_seconds'],'source_snapshot_hashes':{x.name:sha256_file(x) for x in (root/'source_snapshot').iterdir() if x.is_file()}}
    if kind=='vector' and cfg['paired_ising_run_id']!=results[f'{ia}:{seed}']['run_id']:raise ValueError('incorrect paired source')
 rng=np.random.default_rng(22140);boot=rng.integers(0,len(docids),size=(1000,len(docids)));comparisons={}
 for ia,va in PAIRS:
  diff=np.mean([doc_values[f'{ia}:{s}']-doc_values[f'{va}:{s}'] for s in SEEDS],axis=0)
  comparisons[f'{ia}_minus_{va}']={'micro_auc_per_seed':[results[f'{ia}:{s}']['metrics']['noise_pairwise_auc']-results[f'{va}:{s}']['metrics']['noise_pairwise_auc'] for s in SEEDS],
    'macro_auc_difference':float(diff.mean()),'conditional_article_bootstrap_95':[float(x) for x in np.percentile(diff[boot].mean(axis=1),[2.5,97.5])]}
 return {'dataset_run_id':DATA,'dataset_hashes':m['hashes'],'frequency_only':m['frequency_only'],'results':results,'comparisons':comparisons,'limitations':'posthoc validation; count-matched not matched expressiveness or optimization; bootstrap conditional on 60 articles, samples and seeds; no hardware'}

if __name__=='__main__':
 start=time.monotonic();mlflow.set_tracking_uri(tracking_uri());out=audit();out['elapsed_seconds']=time.monotonic()-start;mlflow.set_experiment('word-energy-wikitext2-matched-vectors')
 contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='wikitext_matched_summary',energy_convention='normalized F2 or 3D dot + candidate prior',variable_encoding='spins or real vectors',beta_temperature_convention='Ising beta1',sampler_schedule='exact paired checkpoint audit',seed=22140,data_hash=DATA,data_split='WikiText2 development validation; test untouched',graph_summary='2 spins per token or 3D vectors',limitations=out['limitations'])
 with mlflow.start_run(run_name='matched-vectors-summary') as run:
  log_contract(mlflow,contract,out)
  for key,v in out['comparisons'].items():mlflow.log_metric(key+'.macro_auc_diff',v['macro_auc_difference'])
  for p in (Path(__file__),ROOT/'notes/WIKITEXT2_MATCHED_VECTOR_PROTOCOL.md'):mlflow.log_artifact(str(p),'source_snapshot')
  print(json.dumps({'run_id':run.info.run_id,'comparisons':out['comparisons'],'results':out['results']},indent=2))
