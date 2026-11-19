"""Audit joint-context runs and compare paired article-level outcomes."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.build_wikitext2_joint_context_data import load,EXP
from scripts.build_wikitext2_document_data import doc_macro
from word_energy.wikitext_joint_context import ARMS,SEEDS,functions
from word_energy.wikitext_joint_evaluation import canonical_scores
from word_energy.text8_lexical_models import pair_metrics
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file

def audit(data_id):
    c=MlflowClient();a,m=load(data_id);exp=c.get_experiment_by_name(EXP);runs=[r for r in c.search_runs([exp.experiment_id],max_results=1000) if r.data.tags.get('arm') in ARMS]
    if len(runs)!=12:raise ValueError('expected 12 runs')
    w=a['validation_windows'];noise=a['validation_noise'];counts=a['counts'];docs=a['validation_document'];docids=np.unique(docs);prior=.25*np.log(counts.astype(float))
    score_fns={arm:functions(arm,prior)[0] for arm in ARMS};results={};init={};doc_values={};sources=None
    for run in runs:
        arm=run.data.tags['arm'];seed=int(run.data.tags['seed']);key=f'{arm}:{seed}'
        if key in results or run.info.status!='FINISHED':raise ValueError('duplicate/unfinished')
        root=artifact_dir(c,run.info.run_id);cfg=json.loads((root/'config.json').read_text(encoding='utf8'))
        if not cfg['run_complete'] or cfg['dataset_run_id']!=data_id or cfg['dataset_hashes']!=m['hashes'] or cfg['offset_sha256']!=sha_array(prior):raise ValueError('dataset/config mismatch')
        expected=a['shuffled_windows'] if arm=='joint_shuffled' else a['train_windows']
        if cfg['train_hash']!=sha_array(expected) or cfg['parameter_count']!=(6000 if arm=='vector' else 6002):raise ValueError('train/hash/count mismatch')
        local={p.name:sha256_file(p) for p in (root/'source_snapshot').iterdir() if p.is_file()}
        if sources is not None and local!=sources:raise ValueError('source differs across arms')
        sources=local
        for rel in ('word_energy/wikitext_joint_context.py','word_energy/wikitext_joint_evaluation.py','word_energy/text8_factorial_models.py','word_energy/wikitext_matched_vectors.py','word_energy/text8_lexical_models.py','word_energy/ising/exact_training.py','scripts/build_wikitext2_joint_context_data.py'):
            p=ROOT/rel
            if local[p.name]!=sha256_file(p):raise ValueError('local source differs from snapshot')
        checkpoints={}
        for stage in ('initial','epoch1','epoch2','epoch3'):
            with np.load(root/'checkpoints'/f'{stage}.npz',allow_pickle=False) as z:p={k:z[k] for k in z.files}
            if {k:sha_array(v) for k,v in p.items()}!=cfg['checkpoint_hashes'][stage]:raise ValueError('checkpoint hash')
            if stage in ('initial','epoch3'):checkpoints[stage]=p
        init[key]=checkpoints['initial'];pos,neg=canonical_scores(checkpoints['epoch3'],score_fns[arm],w,noise)
        metrics=pair_metrics(pos,neg,counts[w[:,0]]);metrics['document_macro_auc']=doc_macro(pos,neg,docs)
        for k in ('noise_pairwise_auc','noise_mrr_at_21','document_macro_auc'):
            if not np.isclose(metrics[k],cfg['metrics']['epoch3'][k],atol=1e-14,rtol=0):raise ValueError('recomputed metric mismatch')
        wins=((pos[:,None]>neg)+.5*(pos[:,None]==neg)).mean(axis=1);doc_values[key]=np.array([wins[docs==d].mean() for d in docids])
        results[key]={'run_id':run.info.run_id,'parameter_count':cfg['parameter_count'],'metrics':metrics,'initial_metrics':cfg['metrics']['initial'],
          'diagnostics':cfg['diagnostics'].get('epoch3',{}),'elapsed_seconds':cfg['elapsed_seconds'],'peak_mib':cfg['metrics']['epoch3']['peak_working_set_mib']}
    for seed in SEEDS:
        x=init[f'joint:{seed}']
        for arm in ('pair_add','joint_shuffled'):
            y=init[f'{arm}:{seed}']
            if set(x)!=set(y) or any(not np.array_equal(v,y[k]) for k,v in x.items()):raise ValueError('paired initialization differs')
    rng=np.random.default_rng(23140);indices=rng.integers(0,len(docids),size=(1000,len(docids)));effects={}
    for other in ('pair_add','vector','joint_shuffled'):
        values=[results[f'joint:{s}']['metrics']['noise_pairwise_auc']-results[f'{other}:{s}']['metrics']['noise_pairwise_auc'] for s in SEEDS]
        d=np.mean([doc_values[f'joint:{s}']-doc_values[f'{other}:{s}'] for s in SEEDS],axis=0)
        effects[f'joint_minus_{other}']={'micro_per_seed':values,'macro_difference':float(d.mean()),'article_bootstrap_95_conditional':[float(x) for x in np.percentile(d[indices].mean(axis=1),[2.5,97.5])]}
    return {'dataset_run_id':data_id,'dataset_hashes':m['hashes'],'source_sha256':m['source_sha256'],'source_hashes':sources,'results':results,'frequency_only':m['frequency_only'],
      'n_validation_docs':len(docids),'effects':effects,'limitations':'posthoc development; 3 seeds share windows and one shuffle; conditional document bootstrap; no sense or physical hardware claim'}

if __name__=='__main__':
    data_id=sys.argv[1];start=time.monotonic();mlflow.set_tracking_uri(tracking_uri());out=audit(data_id);out['elapsed_seconds']=time.monotonic()-start;mlflow.set_experiment(EXP)
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='wikitext_joint_context_summary',energy_convention='normalized F3/F2 or CBOW + candidate center prior',variable_encoding='two-spin or 3D vectors',beta_temperature_convention='Ising beta=1',sampler_schedule='exact canonical checkpoint audit',seed=23140,data_hash=data_id,data_split='WikiText development train/validation; all reserved tests untouched',graph_summary='six-spin star or vector CBOW',limitations=out['limitations'])
    with mlflow.start_run(run_name='joint-context-summary') as run:
        log_contract(mlflow,contract,out)
        for name,x in out['effects'].items():mlflow.log_metric(name+'.macro_auc_difference',x['macro_difference'])
        for p in (Path(__file__),ROOT/'notes/WIKITEXT2_JOINT_CONTEXT_PROTOCOL.md'):mlflow.log_artifact(str(p),'source_snapshot')
        print(json.dumps({'run_id':run.info.run_id,'frequency_only':out['frequency_only'],'effects':out['effects'],'results':out['results']},indent=2))
