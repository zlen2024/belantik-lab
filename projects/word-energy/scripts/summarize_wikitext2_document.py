"""Audit all WikiText2 endpoints and summarize document-aware comparisons."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.build_wikitext2_document_data import load,doc_macro
from word_energy.text8_factorial_models import functions as ising_functions
from word_energy.text8_prior_vector import functions as vector_functions
from word_energy.text8_canonical_eval import canonical_candidate_scores
from word_energy.text8_lexical_models import pair_metrics
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file
ARMS=('tied-frequency','role-frequency','vector-frequency');SEEDS=(17,29,43);MODES=('original','shuffled')

def audit(data_id):
    a,m=load(data_id);c=MlflowClient();exp=c.get_experiment_by_name('word-energy-wikitext2-doc-development')
    runs=[r for r in c.search_runs([exp.experiment_id],max_results=1000) if r.data.tags.get('arm') in ARMS]
    if len(runs)!=18:raise ValueError('expected 18 training jobs')
    pairs=a['validation_pairs'];noise=a['validation_noise'];counts=a['counts'];docs=a['validation_document'];prior=.25*np.log(counts.astype(float));docids=np.unique(docs)
    if len(docids)!=60:raise ValueError('expected validation samples from all 60 articles')
    scorers={arm:(vector_functions(prior)[0] if arm=='vector-frequency' else ising_functions(arm,prior)[0]) for arm in ARMS}
    results={};init={};doc_auc={};snapshot_hashes=None
    for run in runs:
        arm=run.data.tags['arm'];seed=int(run.data.tags['seed']);mode=run.data.tags['mode'];key=f'{arm}:{seed}:{mode}'
        if key in results or run.info.status!='FINISHED':raise ValueError('duplicate/unfinished')
        root=artifact_dir(c,run.info.run_id);cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
        if not cfg['run_complete'] or cfg['dataset_run_id']!=data_id or cfg['dataset_hashes']!=m['hashes'] or cfg['offset_sha256']!=sha_array(prior):raise ValueError('run/data mismatch')
        if cfg['train_pair_hash']!=sha_array(a['train_pairs'] if mode=='original' else a['shuffled_pairs']):raise ValueError('wrong training pairs')
        if cfg['parameter_count']!=({'tied-frequency':6002,'role-frequency':12002,'vector-frequency':8000}[arm]):raise ValueError('parameter count')
        sources={p.name:sha256_file(p) for p in (root/'source_snapshot').iterdir() if p.is_file()}
        if snapshot_hashes is not None and sources!=snapshot_hashes:raise ValueError('source snapshot mismatch')
        snapshot_hashes=sources
        for rel in ('word_energy/text8_prior_vector.py','word_energy/text8_factorial_models.py','word_energy/text8_canonical_eval.py','word_energy/text8_lexical_models.py','word_energy/ising/exact_training.py','scripts/build_wikitext2_document_data.py'):
            p=ROOT/rel
            if sha256_file(p)!=sources[p.name]:raise ValueError('local scorer differs from saved source')
        loaded={}
        for stage in ('initial','epoch1','epoch2','epoch3'):
            with np.load(root/'checkpoints'/f'{stage}.npz',allow_pickle=False) as z:p={k:z[k] for k in z.files}
            if {k:sha_array(v) for k,v in p.items()}!=cfg['checkpoint_hashes'][stage]:raise ValueError('checkpoint mismatch')
            if stage in ('initial','epoch3'):loaded[stage]=p
        init[key]=loaded['initial'];pos,neg=canonical_candidate_scores(loaded['epoch3'],scorers[arm],pairs,noise,symmetric=False)
        metrics=pair_metrics(pos,neg,counts[pairs[:,0]]);metrics['document_macro_auc']=doc_macro(pos,neg,docs)
        for k in ('noise_pairwise_auc','noise_mrr_at_21','document_macro_auc'):
            if not np.isclose(metrics[k],cfg['metrics']['epoch3'][k],atol=1e-14,rtol=0):raise ValueError('metric mismatch')
        wins=((pos[:,None]>neg)+.5*(pos[:,None]==neg)).mean(axis=1)
        doc_auc[key]=np.array([wins[docs==i].mean() for i in docids])
        results[key]={'run_id':run.info.run_id,'metrics':metrics,'elapsed_seconds':cfg['elapsed_seconds'],'peak_mib':cfg['metrics']['epoch3']['peak_working_set_mib'],'parameter_count':cfg['parameter_count']}
    for arm in ARMS:
        for seed in SEEDS:
            x=init[f'{arm}:{seed}:original'];y=init[f'{arm}:{seed}:shuffled']
            if set(x)!=set(y) or any(not np.array_equal(v,y[k]) for k,v in x.items()):raise ValueError('paired initial mismatch')
    baseline=m['frequency_only']; pos=prior[pairs[:,1]];neg=prior[noise]
    wins=((pos[:,None]>neg)+.5*(pos[:,None]==neg)).mean(axis=1);base_doc=np.array([wins[docs==i].mean() for i in docids])
    rng=np.random.default_rng(22140);bootstrap=rng.integers(0,len(docids),size=(1000,len(docids)))
    effects={}
    for arm in ARMS:
        original=[results[f'{arm}:{s}:original']['metrics']['noise_pairwise_auc'] for s in SEEDS]
        shuffled=[results[f'{arm}:{s}:shuffled']['metrics']['noise_pairwise_auc'] for s in SEEDS]
        assoc=np.mean([doc_auc[f'{arm}:{s}:original']-doc_auc[f'{arm}:{s}:shuffled'] for s in SEEDS],axis=0)
        baseline_diff=np.mean([doc_auc[f'{arm}:{s}:original']-base_doc for s in SEEDS],axis=0)
        intervals=lambda d:[float(x) for x in np.percentile(d[bootstrap].mean(axis=1),[2.5,97.5])]
        x={'original_minus_shuffled_micro_per_seed':[x-y for x,y in zip(original,shuffled)],'original_minus_frequency_micro_per_seed':[x-baseline['noise_pairwise_auc'] for x in original],
           'original_minus_shuffled_document_macro':float(assoc.mean()),'document_bootstrap_95_percentile':intervals(assoc),
           'original_minus_frequency_document_macro':float(baseline_diff.mean()),'frequency_document_bootstrap_95_percentile':intervals(baseline_diff)}
        if arm!='vector-frequency':
            x['original_minus_vector_micro_per_seed']=[v-results[f'vector-frequency:{s}:original']['metrics']['noise_pairwise_auc'] for s,v in zip(SEEDS,original)]
            dd=np.mean([doc_auc[f'{arm}:{s}:original']-doc_auc[f'vector-frequency:{s}:original'] for s in SEEDS],axis=0)
            x['original_minus_vector_document_macro']=float(dd.mean());x['vector_document_bootstrap_95_percentile']=intervals(dd)
        effects[arm]=x
    return {'dataset_run_id':data_id,'dataset_hashes':m['hashes'],'source_sha256':m['source_sha256'],'source_snapshots':snapshot_hashes,
      'frequency_only':baseline,'n_sampled_articles':len(docids),'doc_bootstrap':'1000 resamples of 60 validation articles, seed 22140; conditions on trained models and sample; descriptive only',
      'results':results,'effects':effects,'limitations':m['limits']}

if __name__=='__main__':
    data_id=sys.argv[1];start=time.monotonic();mlflow.set_tracking_uri(tracking_uri());out=audit(data_id);out['audit_seconds']=time.monotonic()-start
    mlflow.set_experiment('word-energy-wikitext2-doc-development')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='wikitext2_document_summary',energy_convention='normalized F2 or dot + host prior',variable_encoding='spins or 2D vector',beta_temperature_convention='Ising beta=1',sampler_schedule='canonical exact scoring with document-level analysis',seed=22140,data_hash=';'.join(out['source_sha256'].values()),data_split='official document train/validation; test untouched',graph_summary='4-spin Ising or 2D vector',limitations=out['limitations'])
    with mlflow.start_run(run_name='wikitext2-document-summary') as run:
        log_contract(mlflow,contract,out)
        for arm,e in out['effects'].items():mlflow.log_metric(f'{arm}.original_minus_shuffle_macro',e['original_minus_shuffled_document_macro'])
        mlflow.log_artifact(str(Path(__file__).resolve()),'source_snapshot')
        print(json.dumps({'run_id':run.info.run_id,'baseline':out['frequency_only'],'effects':out['effects'],'results':out['results']},indent=2))
