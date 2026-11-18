"""Train one matched vector or run six sequential frozen jobs."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.build_wikitext2_document_data import load
from scripts.train_wikitext2_document import metrics
from scripts.train_text8_lexical_development import snapshot,log_metrics
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.run_ising_corrected_joint import peak_working_set
from word_energy.ising.exact_training import _imports
from word_energy.wikitext_matched_vectors import initial,functions
from word_energy.text8_canonical_eval import canonical_candidate_scores
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract,log_contract,tracking_uri
DATA='b37da8076fd54f4da389082278e9733a';EXP='word-energy-wikitext2-matched-vectors';ARMS=('shared3','directed3');SEEDS=(17,29,43)

def train(arm,seed):
    start=time.monotonic();a,m=load(DATA);prior=.25*np.log(a['counts'].astype(float));shared=arm=='shared3';jax,jnp,_=_imports()
    p=initial(seed,len(prior),shared);score,loss,step,opt=functions(prior,shared);state=opt.init(p);count=sum(np.asarray(v).size for v in p.values())
    client=MlflowClient();oldexp=client.get_experiment_by_name('word-energy-wikitext2-doc-development');oldarm='tied-frequency' if shared else 'role-frequency'
    old=[r for r in client.search_runs([oldexp.experiment_id],max_results=1000) if r.data.tags.get('arm')==oldarm and r.data.tags.get('seed')==str(seed) and r.data.tags.get('mode')=='original']
    if len(old)!=1 or old[0].info.status!='FINISHED':raise ValueError('missing paired Ising')
    oldroot=artifact_dir(client,old[0].info.run_id);oldcfg=json.loads((oldroot/'config.json').read_text(encoding='utf-8'))
    if not oldcfg['run_complete'] or oldcfg['dataset_hashes']!=m['hashes'] or oldcfg['offset_sha256']!=sha_array(prior):raise ValueError('paired data mismatch')
    if (oldcfg['epochs'],oldcfg['batch'],oldcfg['lr'],oldcfg['clip'])!=(3,256,.01,5.):raise ValueError('paired training mismatch')
    exp=mlflow.set_experiment(EXP)
    if any(r.data.tags.get('arm')==arm and r.data.tags.get('seed')==str(seed) for r in client.search_runs([exp.experiment_id],max_results=1000)):raise ValueError('duplicate run')
    details={'dataset_run_id':DATA,'dataset_hashes':m['hashes'],'arm':arm,'seed':seed,'paired_ising_run_id':old[0].info.run_id,'parameter_count':count,'epochs':3,'batch':256,'lr':.01,'clip':5.,'offset_sha256':sha_array(prior),'metrics':{},'checkpoint_hashes':{},'run_complete':False}
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='wikitext_matched_'+arm,energy_convention='3D dot score + host frequency prior',variable_encoding='real vectors',beta_temperature_convention='not applicable',sampler_schedule='ranking objective, Adam 3 epochs batch256',seed=seed,data_hash=hashlib.sha256(''.join(m['source_sha256'].values()).encode()).hexdigest(),data_split='WikiText2 train/validation articles; reserved test untouched',graph_summary='shared or directed 3D vector',limitations='posthoc development; parameter count matched not capacity; no THRML/hardware')
    with mlflow.start_run(run_name=f'wikitext-matched-{arm}-seed{seed}') as run:
        mlflow.set_tags({'arm':arm,'seed':str(seed),'dataset_run_id':DATA,'paired_ising_run_id':old[0].info.run_id})
        mlflow.log_params({'arm':arm,'seed':seed,'parameter_count':count,'epochs':3,'batch':256,'lr':.01,'clip':5.,'dataset_run_id':DATA})
        log_contract(mlflow,contract,details)
        for rel in ('scripts/train_wikitext_matched_vectors.py','word_energy/wikitext_matched_vectors.py','scripts/train_wikitext2_document.py','scripts/build_wikitext2_document_data.py','scripts/train_text8_lexical_development.py','word_energy/text8_canonical_eval.py','word_energy/text8_lexical_models.py','word_energy/ising/exact_training.py','notes/WIKITEXT2_MATCHED_VECTOR_PROTOCOL.md','tests/test_wikitext_matched_vectors.py'):mlflow.log_artifact(str(ROOT/rel),'source_snapshot')
        z={k:jnp.zeros_like(v) for k,v in p.items()};zp,zn=canonical_candidate_scores(z,score,a['validation_pairs'],a['validation_noise'],symmetric=False)
        if not np.array_equal(zp,prior[a['validation_pairs'][:,1]]) or not np.array_equal(zn,prior[a['validation_noise']]):raise ValueError('zero control')
        details['zero_control']='all positive/negative scores exactly equal to candidate frequency prior'
        tmp=ROOT/'.tmp';tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as d:
            t=Path(d);details['checkpoint_hashes']['initial']=snapshot(p,'initial',t);details['metrics']['initial']=metrics(p,score,loss,a);log_metrics('initial',details['metrics']['initial'])
            for epoch in range(3):
                total=0.;n=0
                for i in range(0,len(a['train_pairs']),256):
                    ix=a['orders'][epoch,i:i+256];p,state,value=step(p,state,a['train_pairs'][ix],a['train_noise'][epoch,ix]);value=float(value)
                    if not np.isfinite(value):raise FloatingPointError('nonfinite training loss')
                    total+=len(ix)*value;n+=len(ix)
                peak=peak_working_set()
                if peak>2*1024**3:raise MemoryError('2 GiB check')
                stage=f'epoch{epoch+1}';details['checkpoint_hashes'][stage]=snapshot(p,stage,t);out=metrics(p,score,loss,a);out['train_objective_mean']=total/n;out['peak_working_set_mib']=peak/1024**2
                details['metrics'][stage]=out;log_metrics(stage,out);print(json.dumps({'run_id':run.info.run_id,'arm':arm,'seed':seed,'stage':stage,'auc':out['noise_pairwise_auc'],'macro_auc':out['document_macro_auc']}),flush=True)
            details['elapsed_seconds']=time.monotonic()-start;details['run_complete']=True;log_contract(mlflow,contract,details)

def main():
    mlflow.set_tracking_uri(tracking_uri())
    if sys.argv[1]=='batch':
        load(DATA);client=MlflowClient();exp=mlflow.set_experiment(EXP)
        if any(r.data.tags.get('arm') for r in client.search_runs([exp.experiment_id],max_results=1000)):raise ValueError('existing jobs; inspect first')
        env=os.environ.copy();env.update({'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MLFLOW_DISABLE_AGENT_HINT':'1'});env.pop('MLFLOW_RUN_ID',None)
        start=time.monotonic();deadline=start+1800;n=0
        for seed in SEEDS:
            for arm in ARMS:
                rem=deadline-time.monotonic()
                if rem<=0:raise TimeoutError(f'completed {n}/6')
                try:r=subprocess.run([sys.executable,str(Path(__file__).resolve()),arm,str(seed)],cwd=ROOT,env=env,text=True,capture_output=True,timeout=rem)
                except subprocess.TimeoutExpired as exc:
                    runs=[r for r in client.search_runs([exp.experiment_id],max_results=1000) if r.data.tags.get('arm')==arm and r.data.tags.get('seed')==str(seed)]
                    if len(runs)==1 and runs[0].info.status=='RUNNING':client.set_tag(runs[0].info.run_id,'interruption_reason','shared deadline exceeded');client.set_terminated(runs[0].info.run_id,status='KILLED')
                    raise TimeoutError(f'incomplete {arm}/{seed}; captured output in exception') from exc
                if r.stdout:print(r.stdout.strip(),flush=True)
                if r.returncode:raise RuntimeError(f'failed {arm}/{seed}: {r.stderr}')
                n+=1;print(f'COMPLETE {n}/6',flush=True)
    else:
        arm=sys.argv[1];seed=int(sys.argv[2]);
        if arm not in ARMS or seed not in SEEDS:raise ValueError('outside frozen panel')
        train(arm,seed)
if __name__=='__main__':main()
