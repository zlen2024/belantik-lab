"""Train one width/seed using fit arrays only; holdout is not read here."""
from pathlib import Path
import json,os,sys,tempfile,time
os.environ.setdefault('JAX_PLATFORMS','cpu')
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.run_ising_corrected_joint import peak_working_set
from word_energy.ising.exact_training import _imports
from word_energy.spin_capacity_lexical import initial,scores,coupling_values,validate_pair_ids
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file

def main(data_id,arm,seed):
    if arm not in ('ising6','ising8') or seed not in (17,29,43):raise ValueError('invalid arm/seed')
    start=time.monotonic();mlflow.set_tracking_uri(tracking_uri());client=MlflowClient();root=artifact_dir(client,data_id)
    run=client.get_run(data_id)
    if run.info.status!='FINISHED' or run.data.tags.get('model_family')!='prospective article split':raise ValueError('bad data run')
    meta=json.loads((root/'config.json').read_text(encoding='utf8'))
    keys=('train_pairs','train_positions','train_document','train_noise','orders','counts','noise_q')
    with np.load(root/'dataset/dataset.npz',allow_pickle=False) as z:a={k:z[k] for k in keys}
    if any(sha_array(a[k])!=meta['hashes'][k] for k in keys):raise ValueError('fit array hash mismatch')
    pairs=a['train_pairs'];noise=a['train_noise'];orders=a['orders'];counts=a['counts'];n=len(counts)
    validate_pair_ids(pairs,n)
    if pairs.shape!=(10000,2) or noise.shape!=(3,10000,5) or orders.shape!=(3,10000) or np.any(noise<0) or np.any(noise>=n):raise ValueError('invalid fit arrays')
    if any(not np.array_equal(np.sort(order),np.arange(10000)) for order in orders):raise ValueError('invalid order')
    prior=.25*np.log(counts.astype(float));jax,jnp,optax=_imports()
    if jax.default_backend()!='cpu':raise ValueError('CPU required')
    params=initial(arm,seed,n);opt=optax.chain(optax.clip_by_global_norm(5.),optax.adam(.01));state=opt.init(params);d=int(arm[5:])
    def loss(p,pos,neg):
        b=pos.shape[0];bad=jnp.broadcast_to(pos[:,None,:],(b,5,2)).at[:,:,0].set(neg).reshape((-1,2))
        good=scores(p,pos,prior,arm,coupling_mode='redistribute',kernel_method='butterfly')
        false=scores(p,bad,prior,arm,coupling_mode='redistribute',kernel_method='butterfly').reshape((b,5))
        return jnp.mean(jax.nn.softplus(1-good[:,None]+false))
    @jax.jit
    def step(p,s,pos,neg):
        value,grad=jax.value_and_grad(loss)(p,pos,neg);gnorm=optax.global_norm(grad)
        updates,s=opt.update(grad,s,p);return optax.apply_updates(p,updates),s,value,gnorm
    g0=np.asarray(coupling_values(params['interaction'],d,'redistribute'))
    config={'dataset_run_id':data_id,'fit_hashes':{k:meta['hashes'][k] for k in keys},'vocabulary_sha256':meta['vocabulary_sha256'],
      'arm':arm,'seed':seed,'epochs':3,'batch':128,'optimizer':'Adam .01 clip 5','steps_expected':237,'kernel':'butterfly exact','coupling_initial':g0.tolist(),
      'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','JAX_PLATFORMS')},
      'limitations':'one cue; current-run heldout untouched during fitting; no sense, Gaussian, THRML or hardware claim'}
    contract=RunContract('classical','cpu_gpu_simulator',arm+' prospective exact pair','higher normalized log-partition + .25 log count','spins {-1,+1}','beta=1','exact tensor contraction, Adam',seed,meta['source_sha256'],'fit articles only; heldout not read',f'{2*d} spins, {d*d} edges, max degree {d}',config['limitations'])
    mlflow.set_experiment('word-energy-spin8-prospective')
    with mlflow.start_run(run_name=f'{arm}-seed{seed}') as current:
        mlflow.set_tags({'arm':arm,'seed':str(seed),'dataset_run_id':data_id,'holdout_read':'false'})
        log_contract(mlflow,contract,config);updates=0;epochs=[]
        for epoch in range(3):
            total=0.;gnorms=[]
            for start_i in range(0,len(pairs),128):
                idx=orders[epoch,start_i:start_i+128]
                params,state,loss_value,grad_norm=step(params,state,pairs[idx],noise[epoch,idx])
                l=float(loss_value);g=float(grad_norm)
                if not np.isfinite(l) or not np.isfinite(g):raise FloatingPointError('nonfinite loss or gradient')
                total+=l*len(idx);gnorms.append(g);updates+=1
            peak=peak_working_set()
            if peak>2*1024**3:raise MemoryError('2GiB cap')
            if time.monotonic()-start>900:raise TimeoutError('per-run deadline')
            actual=np.asarray(coupling_values(params['interaction'],d,'redistribute'))
            if not np.isclose(actual.sum(),.8,atol=1e-12,rtol=0) or np.any(actual<=0) or np.any(actual>=1.5/d):raise ValueError('W constraint')
            e={'loss':total/len(pairs),'gradient_norm_max':max(gnorms),'gradient_norm_min':min(gnorms),'W':actual.tolist(),'peak_mib':peak/1024**2};epochs.append(e)
            mlflow.log_metric('train_loss',e['loss'],step=epoch+1);mlflow.log_metric('gradient_norm_max',e['gradient_norm_max'],step=epoch+1)
            print(json.dumps({'run':current.info.run_id,'arm':arm,'seed':seed,'epoch':epoch+1,'train_loss':e['loss'],'peak_mib':e['peak_mib']}),flush=True)
        if updates!=237 or any(not np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(params)):raise ValueError('bad training outcome')
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as dtmp:
            ck=Path(dtmp)/'checkpoint.npz';np.savez_compressed(ck,**{k:np.asarray(v) for k,v in params.items()})
            config['checkpoint_sha256']=sha256_file(ck);mlflow.log_artifact(str(ck),'results')
        config.update({'epochs_detail':epochs,'updates':updates,'elapsed_seconds':time.monotonic()-start,'run_complete':True})
        log_contract(mlflow,contract,config)
        for f in (Path(__file__),ROOT/'word_energy/spin_capacity_lexical.py',ROOT/'tests/test_spin_capacity_lexical.py',ROOT/'notes/SPIN_6_VS_8_PROSPECTIVE_PROTOCOL.md'):mlflow.log_artifact(str(f),'source_snapshot')
        print('FINISHED',current.info.run_id,flush=True)
if __name__=='__main__':main(sys.argv[1],sys.argv[2],int(sys.argv[3]))
