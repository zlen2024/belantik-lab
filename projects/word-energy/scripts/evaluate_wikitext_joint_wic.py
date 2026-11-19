"""Repeat frozen WiC-development subset using new joint-trained parameters."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.build_wikitext2_joint_context_data import load as load_joint_data
from scripts.evaluate_wic_context_probe import bootstrap
from word_energy.wic_context_probe import star,negative_js,cosine,auc
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file
REFERENCE='fa6a014eefc549338b7735a9e52206ed';EXP='word-energy-wikitext2-joint-context';SEEDS=(17,29,43)

def paired_bootstrap(labels,groups,a,b):
    labels=np.asarray(labels);groups=np.asarray(groups);targets=np.unique(groups);idx=np.searchsorted(targets,groups);pos=np.flatnonzero(labels==1);neg=np.flatnonzero(labels==0)
    rng=np.random.default_rng(22150);draws=rng.integers(0,len(targets),size=(1000,len(targets)))
    def matrix(x):return (x[pos,None]>x[None,neg])+.5*(x[pos,None]==x[None,neg])
    aa=[matrix(np.asarray(x)) for x in a];bb=[matrix(np.asarray(x)) for x in b]
    if len(bb)==1:bb=bb*len(aa)
    if len(aa)!=len(bb):raise ValueError('seed count mismatch')
    out=[]
    for draw in draws:
        weights=np.bincount(draw,minlength=len(targets))[idx];wp=weights[pos];wn=weights[neg];denom=wp.sum()*wn.sum()
        if denom:out.append(float(np.mean([(wp@x@wn-wp@y@wn)/denom for x,y in zip(aa,bb)])))
    return {'mean_difference':float(np.mean([auc(labels,x)-auc(labels,y) for x,y in zip(a,b if len(b)>1 else b*len(a))])),
      'interval_95':[float(x) for x in np.percentile(out,[2.5,97.5])],'valid_draws':len(out)}

def audit(data_id):
    c=MlflowClient();root=artifact_dir(c,REFERENCE);ref=json.loads((root/'config.json').read_text(encoding='utf8'))
    _,joint_meta=load_joint_data(data_id)
    if joint_meta['base_hashes']!=ref['dataset_hashes']:raise ValueError('reference vocabulary/base dataset mismatch')
    for rel in ('word_energy/wic_context_probe.py','scripts/evaluate_wic_context_probe.py'):
        p=ROOT/rel
        if sha256_file(p)!=sha256_file(root/'source_snapshot'/p.name):raise ValueError('reference readout source mismatch')
    selected=ref['selected'];labels=np.array(ref['labels'],np.int32);groups=[x['word'] for x in selected]
    if len(selected)!=167 or len(labels)!=167 or int(labels.sum())!=82:raise ValueError('reference subset changed')
    exp=c.get_experiment_by_name(EXP);records=c.search_runs([exp.experiment_id],max_results=1000)
    predictions={};sources={}
    for arm,stages in [('joint',('initial','epoch3')),('pair_add',('epoch3',)),('joint_shuffled',('epoch3',)),('vector',('initial','epoch3'))]:
        for stage in stages:
            scores=[];entropy=[];context_scores=[]
            for seed in SEEDS:
                rows=[r for r in records if r.data.tags.get('arm')==arm and r.data.tags.get('seed')==str(seed)]
                if len(rows)!=1 or rows[0].info.status!='FINISHED':raise ValueError('missing run')
                run=rows[0];rroot=artifact_dir(c,run.info.run_id);cfg=json.loads((rroot/'config.json').read_text(encoding='utf8'))
                if not cfg['run_complete'] or cfg['dataset_run_id']!=data_id:raise ValueError('data mismatch')
                with np.load(rroot/'checkpoints'/f'{stage}.npz',allow_pickle=False) as z:p={k:z[k] for k in z.files}
                if {k:sha_array(v) for k,v in p.items()}!=cfg['checkpoint_hashes'][stage]:raise ValueError('checkpoint mismatch')
                source_key=f'{arm}:{stage}:{seed}';sources[source_key]=run.info.run_id;values=[];ent=[];ctx_only=[]
                for item in selected:
                    t=item['target'];c1=item['context1'];c2=item['context2']
                    if arm=='vector':
                        v=p['shared'];base1=np.mean(v[c1],axis=0);base2=np.mean(v[c2],axis=0);x=v[t]+base1;y=v[t]+base2;values.append(cosine(x,y));ctx_only.append(cosine(base1,base2))
                    else:
                        x=star(p,t,c1,False);y=star(p,t,c2,False);values.append(negative_js(x,y));ent.extend([-np.sum(x*np.log(x)),-np.sum(y*np.log(y))])
                scores.append(values)
                if ent:entropy.append(float(np.mean(ent)))
                if ctx_only:context_scores.append(ctx_only)
            predictions[f'{arm}:{stage}']={'scores':scores,'posterior_entropy':entropy}
            if context_scores:predictions[f'{arm}:{stage}:context_only']={'scores':context_scores}
    for name in ('tied-frequency:original','role-frequency:original','bag_jaccard'):
        predictions['previous:'+name]=ref['predictions'][name]
    summaries={}
    for name,item in predictions.items():
        vectors=[np.asarray(s,dtype=float) for s in item['scores']]
        if any(not np.isfinite(v).all() for v in vectors):raise ValueError('nonfinite')
        scores=[auc(labels,v) for v in vectors];summaries[name]={'auc_per_seed':scores,'auc_mean':float(np.mean(scores)),'bootstrap':bootstrap(labels,groups,vectors)}
    joint=predictions['joint:epoch3']['scores'];effects={}
    for name in ('pair_add:epoch3','joint:initial','joint_shuffled:epoch3','vector:epoch3','previous:bag_jaccard','previous:tied-frequency:original'):
        effects['joint_minus_'+name]=paired_bootstrap(labels,groups,joint,predictions[name]['scores'])
    return {'dataset_run_id':data_id,'wic_reference_run':REFERENCE,'wic_sha256':ref['wic_sha256'],'selected_count':len(selected),'total_examples':ref['total_examples'],
      'positive':int(labels.sum()),'negative':int(len(labels)-labels.sum()),'target_count':len(set(groups)),'source_runs':sources,'predictions':predictions,'summaries':summaries,'paired_effects':effects,
      'limitations':'repeated previously inspected WiC development, 167 selected examples; untrained readout, variable context cardinality, target/topic confounds; not unseen test or hardware'}

if __name__=='__main__':
    data_id=sys.argv[1];start=time.monotonic();mlflow.set_tracking_uri(tracking_uri());out=audit(data_id);out['elapsed_seconds']=time.monotonic()-start;mlflow.set_experiment(EXP)
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='wikitext_joint_repeated_wic_probe',energy_convention='normalized Ising star posterior or shared vector cosine',variable_encoding='two-spin or 3D vector',beta_temperature_convention='Ising beta=1',sampler_schedule='exact posterior; no WiC training',seed=22150,data_hash=out['wic_sha256'],data_split='previously inspected WiC validation subset; no tests',graph_summary='2 spins per token, 1-4 context spokes',limitations=out['limitations'])
    with mlflow.start_run(run_name='joint-repeated-wic-development') as run:
        log_contract(mlflow,contract,out)
        for k,v in out['summaries'].items():mlflow.log_metric(k.replace(':','.')+'.auc_mean',v['auc_mean'])
        for p in (Path(__file__),ROOT/'word_energy/wic_context_probe.py',ROOT/'scripts/evaluate_wic_context_probe.py',ROOT/'scripts/build_wikitext2_joint_context_data.py',ROOT/'notes/WIKITEXT2_JOINT_WIC_PROBE_PROTOCOL.md'):mlflow.log_artifact(str(p),'source_snapshot')
        print(json.dumps({'run_id':run.info.run_id,'selected':out['selected_count'],'summaries':out['summaries'],'paired_effects':out['paired_effects']},indent=2))
