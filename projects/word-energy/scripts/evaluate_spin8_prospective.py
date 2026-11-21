"""Read the prospective article holdout once after all six fits finish."""
from pathlib import Path
import json,sys,tempfile
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from word_energy.ising.exact_training import _imports
from word_energy.spin_capacity_lexical import scores,validate_pair_ids
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file

def main(data_id):
    mlflow.set_tracking_uri(tracking_uri());client=MlflowClient();exp=client.get_experiment_by_name('word-energy-spin8-prospective')
    runs=client.search_runs([exp.experiment_id],max_results=1000)
    selected=[r for r in runs if r.data.tags.get('dataset_run_id')==data_id and r.data.tags.get('arm') in ('ising6','ising8')]
    key=lambda r:(r.data.tags['arm'],int(r.data.tags['seed']))
    expected={(a,s) for a in ('ising6','ising8') for s in (17,29,43)}
    if len(selected)!=6 or {key(r) for r in selected}!=expected or any(r.info.status!='FINISHED' for r in selected):raise ValueError('expected exactly six finished fits')
    checkpoint_params={};checkpoint_ids={};root=artifact_dir(client,data_id);meta=json.loads((root/'config.json').read_text(encoding='utf8'))
    for r in selected:
        rr=artifact_dir(client,r.info.run_id);cfg=json.loads((rr/'config.json').read_text(encoding='utf8'))
        ck=rr/'results/checkpoint.npz'
        if not cfg.get('run_complete') or cfg['updates']!=237 or cfg['vocabulary_sha256']!=meta['vocabulary_sha256'] or sha256_file(ck)!=cfg['checkpoint_sha256']:raise ValueError('fit integrity mismatch')
        with np.load(ck,allow_pickle=False) as f:checkpoint_params[key(r)]={k:f[k] for k in f.files}
        checkpoint_ids[str(key(r))]=r.info.run_id
    needed=('holdout_pairs','holdout_noise','holdout_document','counts')
    with np.load(root/'dataset/dataset.npz',allow_pickle=False) as f:data={k:f[k] for k in needed}
    if any(sha_array(data[k])!=meta['hashes'][k] for k in needed):raise ValueError('holdout hash mismatch')
    pairs,noise,docs,counts=(data[k] for k in needed);validate_pair_ids(pairs,len(counts))
    if pairs.shape!=(2000,2) or noise.shape!=(2000,5) or np.any(noise<0) or np.any(noise>=len(counts)):raise ValueError('holdout shape or ID')
    prior=.25*np.log(counts.astype(float));doc_ids=np.unique(docs);jax,jnp,_=_imports()
    if jax.default_backend()!='cpu':raise ValueError('CPU required')
    saved={};doc_scores={}
    def calc(p,arm):
        score=jax.jit(lambda x:scores(p,x,prior,arm,coupling_mode='redistribute',kernel_method='butterfly'))
        allscores=[]
        for i in range(0,len(pairs),128):
            w=pairs[i:i+128];z=noise[i:i+128];batch=np.broadcast_to(w[:,None,:],(len(w),6,2)).copy();batch[:,1:,0]=z
            allscores.append(np.asarray(score(jnp.asarray(batch.reshape(-1,2)))).reshape(len(w),6))
        vals=np.concatenate(allscores)
        if not np.isfinite(vals).all():raise FloatingPointError('nonfinite score')
        collision=noise==pairs[:,0,None]
        if np.any(vals[:,1:][collision]!=np.broadcast_to(vals[:,:1],collision.shape)[collision]):raise ValueError('identical pair scores disagree')
        return vals
    def summarize(vals):
        credits=((vals[:,:1]>vals[:,1:])+.5*(vals[:,:1]==vals[:,1:])).mean(axis=1)
        perdoc=np.array([credits[docs==doc].mean() for doc in doc_ids]);return credits,perdoc
    baseline=np.column_stack((prior[pairs[:,0]],prior[noise]));bp,bd=summarize(baseline)
    saved['prior_scores']=baseline;doc_scores['prior']=bd
    summary={'dataset_run_id':data_id,'fit_runs':checkpoint_ids,'documents':len(doc_ids),'examples':len(pairs),
             'prior':{'pair_auc':float(bp.mean()),'document_macro_auc':float(bd.mean())},'models':{},
             'limitations':'prospective holdout from historical WikiText train only; sampled lexical noise, no sense, Gaussian, THRML or chip claim'}
    for arm,seed in sorted(expected):
        values=calc(checkpoint_params[(arm,seed)],arm);credits,perdoc=summarize(values);name=f'{arm}-{seed}'
        saved[name]=values;doc_scores[name]=perdoc
        summary['models'][name]={'pair_auc':float(credits.mean()),'document_macro_auc':float(perdoc.mean()),
          'positive_std':float(values[:,0].std()),'tie_fraction':float(np.mean(values[:,:1]==values[:,1:]))}
    draws=np.random.default_rng(89130).integers(len(doc_ids),size=(2000,len(doc_ids)))
    comparisons={}
    for label,other in [('ising8_minus_ising6','ising6'),('ising8_minus_prior','prior'),('ising6_minus_prior','prior')]:
        arm='ising6' if label.startswith('ising6') else 'ising8'
        deltas=np.stack([doc_scores[f'{arm}-{s}']-(doc_scores[other] if other=='prior' else doc_scores[f'{other}-{s}']) for s in (17,29,43)])
        pooled=deltas.mean(axis=0);boot=pooled[draws].mean(axis=1)
        comparisons[label]={'mean_document_delta':float(pooled.mean()),'per_seed':deltas.mean(axis=1).tolist(),'ci95':np.percentile(boot,[2.5,97.5]).tolist()}
    summary['comparisons']=comparisons
    summary['advance_gate']=all(all(v>0 for v in comparisons[k]['per_seed']) and comparisons[k]['ci95'][0]>0 for k in ('ising8_minus_ising6','ising8_minus_prior'))
    mlflow.set_experiment('word-energy-spin8-prospective')
    limits=summary['limitations']
    with mlflow.start_run(run_name='spin6-vs-spin8-reserved-article-evaluation') as run:
        log_contract(mlflow,RunContract('classical','cpu_gpu_simulator','6 vs 8 exact lexical evaluation','higher normalized pair log partition + prior','spins {-1,+1}','beta=1','exact tensor contraction',89130,meta['source_sha256'],'prospectively heldout source-train articles','two tokens: 12/16 spins, degree 6/8',limits),summary)
        mlflow.log_dict(summary,'summary.json')
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as d:
            p=Path(d)/'holdout_scores.npz';np.savez_compressed(p,**saved,document=docs);mlflow.log_artifact(str(p),'results')
        for name,x in summary['models'].items():mlflow.log_metric(name+'.document_macro_auc',x['document_macro_auc'])
        mlflow.log_metric('prior.document_macro_auc',summary['prior']['document_macro_auc'])
        for path in (Path(__file__),ROOT/'notes/SPIN_6_VS_8_PROSPECTIVE_PROTOCOL.md',ROOT/'word_energy/spin_capacity_lexical.py',ROOT/'tests/test_spin_capacity_lexical.py'):mlflow.log_artifact(str(path),'source_snapshot')
        print('RUN',run.info.run_id,json.dumps(summary),flush=True)
if __name__=='__main__':main(sys.argv[1])
