"""Matched fixed-vs-learned total W; reads WikiText source-fit arrays only."""
from itertools import combinations
from pathlib import Path
import json, os, sys, tempfile, time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.run_ising_corrected_joint import peak_working_set
from word_energy.ising.exact_training import _imports
from word_energy.spin_capacity_lexical import initial, states_for, diagonal_kernel_apply, validate_pair_ids
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract, log_contract, tracking_uri, sha256_file

DATA = '3387fc4fb99646f1b7aacd518f39bf10'

def main(seed):
    if seed not in (17, 29, 43): raise ValueError('seed')
    mlflow.set_tracking_uri(tracking_uri()); client = MlflowClient()
    source = artifact_dir(client, DATA)
    meta = json.loads((source/'config.json').read_text(encoding='utf8'))
    keys = ('train_pairs', 'train_noise', 'orders', 'counts')
    with np.load(source/'dataset/dataset.npz', allow_pickle=False) as z: data = {k:z[k] for k in keys}
    if any(sha_array(data[k]) != meta['hashes'][k] for k in keys): raise ValueError('fit hash')
    pairs, noise, orders, counts = (data[k] for k in keys)
    validate_pair_ids(pairs, len(counts))
    if pairs.shape != (10000,2) or noise.shape != (3,10000,5) or orders.shape != (3,10000): raise ValueError('shape')
    if np.any(noise < 0) or np.any(noise >= len(counts)) or any(not np.array_equal(np.sort(x),np.arange(10000)) for x in orders): raise ValueError('noise or order')
    jax,jnp,optax = _imports()
    if jax.default_backend() != 'cpu': raise ValueError('CPU required')
    d=6; s=states_for(d,jnp); prods=jnp.stack([s[:,i]*s[:,j] for i,j in combinations(range(d),2)],axis=1)
    prior=jnp.asarray(.25*np.log(counts.astype(float)))
    p0=initial('ising6',seed,len(counts)); p0['interaction']=jnp.zeros((d,),dtype=jnp.float64); p0['strength']=jnp.asarray(np.log(.8/.7),dtype=jnp.float64)
    def strength(p): return 1.5*jax.nn.sigmoid(p['strength'])
    def score(p,pair):
        ids=pair.reshape(-1); logits=(2.5*jnp.tanh(p['biases'][ids]))@s.T+(2*jnp.tanh(p['internal'][ids]))@prods.T
        q=jax.nn.softmax(logits,axis=1).reshape((len(pair),2,1<<d))
        g=strength(p)*jax.nn.softmax(p['interaction'])
        return jnp.log(jnp.sum(q[:,0,:]*diagonal_kernel_apply(q[:,1,:],g),axis=1))+prior[pair[:,0]]
    def loss(p,pos,neg):
        b=pos.shape[0]; bad=jnp.broadcast_to(pos[:,None,:],(b,5,2)).at[:,:,0].set(neg).reshape((-1,2))
        return jnp.mean(jax.nn.softplus(1-score(p,pos)[:,None]+score(p,bad).reshape((b,5))))
    test_idx=orders[0,:128]; pos=jnp.asarray(pairs[test_idx]); neg=jnp.asarray(noise[0,test_idx])
    val,grad=jax.value_and_grad(loss)(p0,pos,neg); eps=1e-5
    twin=jax.tree_util.tree_map(lambda x:x.copy(),p0)
    twin_val,twin_grad=jax.value_and_grad(loss)(twin,pos,neg)
    np.testing.assert_array_equal(np.asarray(score(p0,pos)),np.asarray(score(twin,pos)))
    np.testing.assert_array_equal(np.asarray(val),np.asarray(twin_val))
    for name in ('biases','internal','interaction'):
        np.testing.assert_array_equal(np.asarray(grad[name]),np.asarray(twin_grad[name]))
    plus=dict(p0);minus=dict(p0);plus['strength']=p0['strength']+eps;minus['strength']=p0['strength']-eps
    finite=(float(loss(plus,pos,neg))-float(loss(minus,pos,neg)))/(2*eps)
    np.testing.assert_allclose(float(grad['strength']),finite,rtol=1e-4,atol=1e-8)
    assert np.isfinite(float(val)) and np.isclose(float(strength(p0)),.8,atol=1e-12)
    for arm in ('fixed','learned'):
        started=time.monotonic(); p=jax.tree_util.tree_map(lambda x:x.copy(),p0)
        optimizer=optax.chain(optax.clip_by_global_norm(5.),optax.adam(.01)); state=optimizer.init(p)
        @jax.jit
        def step(p,st,pos,neg):
            value,g=jax.value_and_grad(loss)(p,pos,neg); full=optax.global_norm(g); ag=jnp.abs(g['strength']); rg=optax.global_norm(g['interaction'])
            if arm=='fixed': g=dict(g);g['strength']=jnp.zeros_like(g['strength'])
            upd,st=optimizer.update(g,st,p); return optax.apply_updates(p,upd),st,value,full,ag,rg
        config={'data_run':DATA,'seed':seed,'arm':arm,'fit_hashes':{k:meta['hashes'][k] for k in keys},'vocabulary_sha256':meta['vocabulary_sha256'], 'initial_lambda':.8,'finite_difference_strength':finite,'autodiff_strength':float(grad['strength']),'epochs':3,'batch':128,'notes':'classical lexical pair fit; no sense evaluation, no hardware'}
        contract=RunContract('classical','cpu_gpu_simulator','matched ising6 W strength','normalized pair log partition plus .25 log count','spins {-1,+1}','beta=1','exact butterfly and Adam',seed,meta['source_sha256'],'WikiText source fit only', '12 spins, diagonal cross edges, 6 spins/word',config['notes'])
        mlflow.set_experiment('word-energy-learnable-w-strength')
        with mlflow.start_run(run_name=f'{arm}-seed{seed}') as run:
            mlflow.set_tags({'arm':arm,'seed':str(seed),'dataset_run_id':DATA,'holdout_read':'false'});log_contract(mlflow,contract,config)
            history=[]; updates=0
            for epoch in range(3):
                totals=[]; norms=[]
                for start in range(0,10000,128):
                    idx=orders[epoch,start:start+128];p,state,value,full,ag,rg=step(p,state,pairs[idx],noise[epoch,idx]); row=[float(value),float(full),float(ag),float(rg)]
                    if not np.isfinite(row).all():raise FloatingPointError('nonfinite')
                    totals.append((row[0],len(idx)));norms.append(row[1:]);updates+=1
                if peak_working_set()>2*1024**3 or time.monotonic()-started>900:raise RuntimeError('compute budget')
                lam=float(strength(p));g=np.asarray(lam*jax.nn.softmax(p['interaction']))
                if not (0<lam<1.5) or np.any(g<=0) or (arm=='fixed' and abs(lam-.8)>1e-12):raise ValueError('coupling bounds')
                e={'loss':sum(x*n for x,n in totals)/10000,'lambda':lam,'g':g.tolist(),'gradient_norm_max':np.max(norms,axis=0).tolist()};history.append(e)
                mlflow.log_metric('train_loss',e['loss'],step=epoch+1);mlflow.log_metric('lambda',lam,step=epoch+1);mlflow.log_metric('strength_gradient_max',e['gradient_norm_max'][1],step=epoch+1)
                print(json.dumps({'run':run.info.run_id,'arm':arm,'seed':seed,'epoch':epoch+1,**e}),flush=True)
            if updates!=237 or any(not np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(p)):raise ValueError('training output')
            with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as t:
                path=Path(t)/'checkpoint.npz';np.savez_compressed(path,**{k:np.asarray(v) for k,v in p.items()}); config['checkpoint_sha256']=sha256_file(path);mlflow.log_artifact(str(path),'results')
            config.update({'history':history,'updates':updates,'elapsed_seconds':time.monotonic()-started,'run_complete':True});log_contract(mlflow,contract,config)
            for f in (Path(__file__),ROOT/'notes/LEARNABLE_W_STRENGTH_PROTOCOL.md',ROOT/'word_energy/spin_capacity_lexical.py'):mlflow.log_artifact(str(f),'source_snapshot')
            print('FINISHED',run.info.run_id,flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
