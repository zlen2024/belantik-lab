"""One paired WikiText2 model/seed/context-order development run."""
from pathlib import Path
import hashlib,json,os,sys,tempfile,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.build_wikitext2_document_data import load,doc_macro
from scripts.train_text8_lexical_development import snapshot,log_metrics
from scripts.train_text8_frequency_shuffle import zero_control
from scripts.run_ising_corrected_joint import peak_working_set
from word_energy.ising.exact_training import _imports
from word_energy.text8_factorial_models import functions as ising_functions,initial as ising_initial
from word_energy.text8_prior_vector import functions as vector_functions,initial as vector_initial
from word_energy.text8_canonical_eval import canonical_candidate_scores
from word_energy.text8_lexical_models import pair_metrics
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract,log_contract,tracking_uri
ARMS=('tied-frequency','role-frequency','vector-frequency');SEEDS=(17,29,43)

def metrics(params,score,loss,a):
    pairs=a['validation_pairs'];noise=a['validation_noise'];pos,neg=canonical_candidate_scores(params,score,pairs,noise,symmetric=False)
    result=pair_metrics(pos,neg,a['counts'][pairs[:,0]]);result['document_macro_auc']=doc_macro(pos,neg,a['validation_document'])
    weighted=0.
    for i in range(0,len(pairs),500):
        end=min(i+500,len(pairs));weighted+=float(loss(params,pairs[i:end],noise[i:end,:5]))*(end-i)
    result['objective_loss']=weighted/len(pairs);return result

if __name__=='__main__':
    data_id,arm,seed,mode=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
    if arm not in ARMS or seed not in SEEDS or mode not in ('original','shuffled'):raise ValueError('outside fixed panel')
    start=time.monotonic();mlflow.set_tracking_uri(tracking_uri());a,m=load(data_id);offset=.25*np.log(a['counts'].astype(float))
    vector=arm=='vector-frequency';jax,jnp,_=_imports()
    if vector:params=vector_initial(seed,len(offset));score,loss,step,optimizer=vector_functions(offset)
    else:params=ising_initial(arm,seed,len(offset));score,loss,step,optimizer=ising_functions(arm,offset)
    state=optimizer.init(params);train=a['train_pairs'] if mode=='original' else a['shuffled_pairs'];count=sum(np.asarray(v).size for v in params.values())
    exp=mlflow.set_experiment('word-energy-wikitext2-doc-development')
    matches=[r for r in MlflowClient().search_runs([exp.experiment_id],max_results=1000) if r.data.tags.get('arm')==arm and r.data.tags.get('seed')==str(seed) and r.data.tags.get('mode')==mode]
    if matches:raise ValueError('existing run; inspect before retry')
    source_hash=hashlib.sha256(''.join(m['source_sha256'].values()).encode()).hexdigest()
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='wikitext2_'+arm,
      energy_convention='2D dot + host prior' if vector else 'dimensionless normalized F2 + host prior',variable_encoding='2D real vectors' if vector else 'spins {-1,+1}',
      beta_temperature_convention='not applicable' if vector else 'beta=1',sampler_schedule='exact scoring, fixed noise, Adam 3 epochs batch256',seed=seed,data_hash=source_hash,
      data_split='WikiText2 official document-separated train/validation; tests untouched',graph_summary='directed 2D vector' if vector else '4 spins, 4 edges, max degree 2',limitations=m['limits'])
    details={'dataset_run_id':data_id,'dataset_hashes':m['hashes'],'arm':arm,'seed':seed,'mode':mode,'train_pair_hash':sha_array(train),
      'parameter_count':count,'offset_sha256':sha_array(offset),'epochs':3,'batch':256,'lr':.01,'clip':5.,'checkpoint_rule':'epoch3',
      'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},'metrics':{},'checkpoint_hashes':{},'run_complete':False}
    with mlflow.start_run(run_name=f'wikitext2-{arm}-{mode}-seed{seed}') as run:
        mlflow.set_tags({'arm':arm,'seed':str(seed),'mode':mode,'dataset_run_id':data_id,'test_status':'reserved not evaluated'})
        mlflow.log_params({'arm':arm,'seed':seed,'mode':mode,'parameter_count':count,'epochs':3,'batch':256,'lr':.01,'clip':5.,'dataset_run_id':data_id})
        log_contract(mlflow,contract,details)
        for rel in ('scripts/train_wikitext2_document.py','scripts/build_wikitext2_document_data.py','scripts/run_wikitext2_document.py','word_energy/text8_factorial_models.py','word_energy/text8_prior_vector.py','word_energy/text8_canonical_eval.py','word_energy/text8_lexical_models.py','word_energy/ising/exact_training.py','scripts/train_text8_frequency_shuffle.py','scripts/train_text8_lexical_development.py','notes/WIKITEXT2_DOCUMENT_DEVELOPMENT_PROTOCOL.md','tests/test_wikitext2_document_data.py'):
            mlflow.log_artifact(str(ROOT/rel),'source_snapshot')
        if vector:
            zeros={k:jnp.zeros_like(v) for k,v in params.items()};zp,zn=canonical_candidate_scores(zeros,score,a['validation_pairs'],a['validation_noise'],symmetric=False)
            if not np.array_equal(zp,offset[a['validation_pairs'][:,1]]) or not np.array_equal(zn,offset[a['validation_noise']]):raise ValueError('zero-vector scores differ')
            control=pair_metrics(zp,zn,a['counts'][a['validation_pairs'][:,0]])
        else:control=zero_control(params,arm,offset,a,m['frequency_only'])
        for k in ('noise_pairwise_auc','noise_mrr_at_21'):
            if not np.isclose(control[k],m['frequency_only'][k],rtol=0,atol=1e-14):raise ValueError('zero control mismatch')
        details['zero_control']=control;tmp=ROOT/'.tmp';tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as d:
            p=Path(d);details['checkpoint_hashes']['initial']=snapshot(params,'initial',p);details['metrics']['initial']=metrics(params,score,loss,a);log_metrics('initial',details['metrics']['initial'])
            for epoch in range(3):
                total=0.;n=0
                for i in range(0,len(train),256):
                    ix=a['orders'][epoch,i:i+256];params,state,value=step(params,state,train[ix],a['train_noise'][epoch,ix]);value=float(value)
                    if not np.isfinite(value):raise FloatingPointError('nonfinite loss')
                    total+=len(ix)*value;n+=len(ix)
                peak=peak_working_set()
                if peak>2*1024**3:raise MemoryError('2 GiB working set exceeded')
                stage=f'epoch{epoch+1}';details['checkpoint_hashes'][stage]=snapshot(params,stage,p);r=metrics(params,score,loss,a)
                r['train_objective_mean']=total/n;r['peak_working_set_mib']=peak/1024**2;details['metrics'][stage]=r;log_metrics(stage,r)
                print(json.dumps({'run_id':run.info.run_id,'arm':arm,'mode':mode,'seed':seed,'stage':stage,'auc':r['noise_pairwise_auc'],'macro_auc':r['document_macro_auc'],'mrr':r['noise_mrr_at_21']}),flush=True)
            details['elapsed_seconds']=time.monotonic()-start;details['run_complete']=True;log_contract(mlflow,contract,details);mlflow.log_metric('resource.elapsed_seconds',details['elapsed_seconds'])
