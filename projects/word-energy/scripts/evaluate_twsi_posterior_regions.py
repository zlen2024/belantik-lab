"""Evaluate candidate-derived target regions with context, prior and shuffle."""
from pathlib import Path
import json,sys
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.train_twsi_pair_pilot import metrics
from word_energy.posterior_region_probe import posterior_and_regions
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract,log_contract,sha256_file,tracking_uri

PILOT='8f2b8e9cfa8344d6bed4c2155e003cda';SPLIT='16e0bf36c0574dcb92cd88cdf1a9b1d3'
def main():
    mlflow.set_tracking_uri(tracking_uri());client=MlflowClient();pr=artifact_dir(client,PILOT);sr=artifact_dir(client,SPLIT)
    pilot=json.loads((pr/'summary.json').read_text(encoding='utf8'));split=json.loads((sr/'config.json').read_text(encoding='utf8'))
    with np.load(sr/'dataset/split.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    if {k:sha_array(v) for k,v in data.items()}!=split['hashes'] or pilot['split_hashes']!=split['hashes']:raise ValueError('split mismatch')
    with np.load(pr/'dataset/contexts.npz',allow_pickle=False) as f:contexts=f['contexts']
    if sha_array(contexts)!=pilot['contexts_sha256']:raise ValueError('context hash mismatch')
    ix=data['validation_indices'];rows=data['examples'][ix];ctx=contexts[ix]
    shuffled=ctx[np.random.default_rng(50777).permutation(len(ix))]
    result={'pilot_run':PILOT,'split_run':SPLIT,'rows':len(rows),'contexts_sha256':pilot['contexts_sha256'],'seeds':{},
      'limitations':'repeated development weak labels; candidate-derived regions not validated senses; neutral candidate slots; no training/Gaussian/THRML/hardware'}
    predictions={}
    for seed in (17,29,43):
        ck=pr/'checkpoints'/f'ising-seed{seed}.npz'
        if sha256_file(ck)!=pilot['results'][f'ising-{seed}']['checkpoint_sha256']:raise ValueError('checkpoint hash mismatch')
        with np.load(ck,allow_pickle=False) as f:p={k:f[k] for k in f.files}
        post,prior,region,qa,qb=posterior_and_regions(p,rows[:,0],ctx,rows[:,2],rows[:,3])
        shuf,prior2,region2,_,_=posterior_and_regions(p,rows[:,0],shuffled,rows[:,2],rows[:,3])
        neutral,*_=posterior_and_regions(p,rows[:,0],np.full_like(ctx,len(p['biases'])),rows[:,2],rows[:,3])
        np.testing.assert_allclose(neutral,prior,atol=1e-12,rtol=0)
        np.testing.assert_allclose(prior2,prior,atol=1e-12,rtol=0)
        np.testing.assert_array_equal(region2,region)
        if not np.isfinite(post).all() or not np.allclose(post.sum(axis=1),1.,atol=1e-12,rtol=0):raise FloatingPointError('invalid posterior')
        def values(posterior):return np.sum(posterior*(2*region-1),axis=1)
        real,base,shuffle=values(post),values(prior),values(shuf)
        def report(score):return metrics(np.column_stack((score,-score)),rows)
        tv=.5*np.sum(np.abs(post-prior),axis=1)
        info={'real':report(real),'unary':report(base),'shuffled':report(shuffle),
              'mean_target_tv_from_unary':float(tv.mean()),'median_target_tv_from_unary':float(np.median(tv)),
              'mean_abs_region_margin_shift':float(np.mean(np.abs(real-base))),
              'candidate_equal_distribution_rows':int(np.sum(np.all(qa==qb,axis=1))),
              'mean_A_region_state_count_half_ties':float(np.mean(region.sum(axis=1)))}
        result['seeds'][str(seed)]=info
        predictions[str(seed)]={'example_indices':ix.tolist(),'real_margin':real.tolist(),'unary_margin':base.tolist(),
                                'shuffled_margin':shuffle.tolist(),'target_tv':tv.tolist()}
    mlflow.set_experiment('word-energy-natural-context-evaluation')
    with mlflow.start_run(run_name='twsi-target-posterior-region-diagnostic') as run:
        contract=RunContract('classical','cpu_gpu_simulator','four-spin target posterior region diagnostic',
         'candidate-derived region mass margin; higher ranks A','four spins {-1,+1}','beta=1','exact factorized marginal',50777,
         split['hashes']['examples'],'reused TWSI training-derived validation','target and four contexts, two uniform candidate slots; 28 spins max degree 9',result['limitations'])
        log_contract(mlflow,contract,result);mlflow.log_dict(result,'summary.json');mlflow.log_dict(predictions,'predictions.json')
        for seed,info in result['seeds'].items():
            for cond in ('real','unary','shuffled'):
                mlflow.log_metric(f'seed{seed}.{cond}.pair_accuracy',info[cond]['pair_accuracy'])
                mlflow.log_metric(f'seed{seed}.{cond}.both_correct',info[cond]['bidirectional_all_strict_accuracy'])
            mlflow.log_metric(f'seed{seed}.mean_target_tv',info['mean_target_tv_from_unary'])
        for p in (Path(__file__),ROOT/'word_energy/posterior_region_probe.py',ROOT/'tests/test_posterior_region_probe.py',ROOT/'notes/TWSI_POSTERIOR_REGION_DIAGNOSTIC_PROTOCOL.md'):mlflow.log_artifact(str(p),'source_snapshot')
        print('RUN',run.info.run_id,json.dumps(result),flush=True)
if __name__=='__main__':main()
