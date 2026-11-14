"""Run the preregistered reciprocal weighted-family-mean bias transfer."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mlflow
from scripts.analyze_ising_anchor_histories import unary_all
from scripts.diagnose_ising_validation import diagnose_raw
from word_energy.ising.exact_training import ExactTrainingConfig, _free_energy_factory, _lexical_auc, prepare_data, train_smoke
from word_energy.synthetic import POOLS
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri


def load_initial(seed):
    path = ROOT / 'experiments' / 'runs' / f'ising-attribution-original-seed{seed}.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    if record['reproduction_pass'] is not True:
        raise ValueError('saved replay did not pass')
    return path, {k: np.asarray(v, dtype=np.float64) for k, v in record['snapshots']['initial_0'].items()}


def transfer(background, donor, data):
    raw = {k: v.copy() for k, v in background.items()}
    counts = Counter(int(token) for row in data['train_windows'] for token in row[1:])
    report = {}
    all_ids = []
    for family in ('F', 'R'):
        ids = [data['ids'][token] for token in POOLS[family]]
        all_ids.extend(ids)
        weights = np.asarray([counts[i] for i in ids], dtype=np.float64)
        old_mean = np.average(background['biases'][ids], axis=0, weights=weights)
        donor_mean = np.average(donor['biases'][ids], axis=0, weights=weights)
        raw['biases'][ids] = background['biases'][ids] + donor_mean - old_mean
        actual_mean = np.average(raw['biases'][ids], axis=0, weights=weights)
        mean_error = float(np.max(np.abs(actual_mean - donor_mean)))
        residual_error = float(np.max(np.abs((raw['biases'][ids] - actual_mean) - (background['biases'][ids] - old_mean))))
        if mean_error > 1e-12 or residual_error > 1e-12:
            raise ValueError('family mean or token residual mismatch')
        report[family] = {'train_occurrence_weights': weights.tolist(), 'background_mean': old_mean.tolist(), 'donor_mean': donor_mean.tolist(), 'actual_mean': actual_mean.tolist(), 'mean_error': mean_error, 'residual_error': residual_error}
    outside = ~np.isin(np.arange(len(raw['biases'])), all_ids)
    if not np.array_equal(raw['biases'][outside], background['biases'][outside]) or any(not np.array_equal(raw[k], background[k]) for k in ('internal', 'interaction')):
        raise ValueError('unexpected parameter changed')
    report['raw_bias_range'] = [float(raw['biases'][all_ids].min()), float(raw['biases'][all_ids].max())]
    report['transformed_bias_range'] = [float((2.5*np.tanh(raw['biases'][all_ids])).min()), float((2.5*np.tanh(raw['biases'][all_ids])).max())]
    return raw, report


if __name__ == '__main__':
    background_seed, donor_seed = int(sys.argv[1]), int(sys.argv[2])
    if (background_seed, donor_seed) not in ((29,43),(43,29)):
        raise ValueError('only two preregistered reciprocal transfers allowed')
    backpath, background = load_initial(background_seed)
    donorpath, donor = load_initial(donor_seed)
    data = prepare_data()
    initial, checks = transfer(background, donor, data)
    snapshots = {}
    def capture(stage, step, values):
        if not all(np.isfinite(v).all() for v in values.values()): raise FloatingPointError(f'{stage} {step}')
        snapshots[f'{stage}_{step}'] = {k:v.tolist() for k,v in values.items()}
    started=time.monotonic()
    _, metrics = train_smoke(ExactTrainingConfig(seed=background_seed, joint_updates=0), snapshot_callback=capture, initial_override=initial)
    elapsed=time.monotonic()-started
    fixed = {k:np.asarray(v, dtype=np.float64) for k,v in snapshots['lexical_400'].items()}
    digest=hashlib.sha256(json.dumps(snapshots['lexical_400'], sort_keys=True).encode()).hexdigest()
    diag=diagnose_raw(fixed,background_seed,'family-mean-fixed-lexical400',digest)
    p=unary_all(fixed); ids=data['ids']
    fs=p[[ids[t] for t in POOLS['F']]]; rs=p[[ids[t] for t in POOLS['R']]]
    def within(x): return float(np.mean([np.abs(x[i]-x[j]).sum()/2 for i in range(len(x)) for j in range(i+1,len(x))]))
    geometry={'unweighted_anchor_tv':float(np.abs(fs.mean(axis=0)-rs.mean(axis=0)).sum()/2),'within_F_tv':within(fs),'within_R_tv':within(rs),'cross_family_mean_tv':float(np.mean([np.abs(a-b).sum()/2 for a in fs for b in rs])),'per_token_state_probabilities':{t:p[ids[t]].tolist() for f in ('F','R') for t in POOLS[f]}}
    jax,jnp,energy,_,_=_free_energy_factory()
    def lexical(pairs, neg):
        pos=jax.vmap(lambda pair:energy(fixed,pair))(pairs)
        negative=jax.vmap(lambda left,row:jax.vmap(lambda right:energy(fixed,jnp.asarray([left,right],dtype=jnp.int32)))(row))(pairs[:,0],neg)
        return float(jnp.mean(jax.nn.softplus(1+pos[:,None]-negative)))
    val_loss=lexical(jnp.asarray(data['validation_pairs']),jnp.asarray(data['validation_negatives']))
    train_loss=lexical(jnp.asarray(data['train_pairs']),jnp.asarray(data['train_negatives']))
    auc=_lexical_auc(fixed,energy,data,jnp)
    chosen={k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in metrics.items()}
    payload={'background_seed':background_seed,'donor_seed':donor_seed,'source_background_sha256':sha256_file(backpath),'source_donor_sha256':sha256_file(donorpath),'runner_sha256':sha256_file(Path(__file__).resolve()),'training_source_sha256':sha256_file(ROOT/'word_energy/ising/exact_training.py'),'thread_settings':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},'memory_peak':'not measured','checks':checks,'elapsed_seconds':elapsed,'selected_metrics':chosen,'fixed_400_diagnosis':diag,'fixed_400_geometry':geometry,'validation_lexical_loss':val_loss,'full_train_lexical_loss':train_loss,'validation_lexical_macro_auc':auc,'snapshots':snapshots}
    mlflow.set_tracking_uri(tracking_uri()); mlflow.set_experiment('word-energy-exact-training')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='ising_family_mean_bias_transfer',energy_convention='dimensionless beta E; normalized Ising',variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',sampler_schedule='exact enumeration; 400 lexical optimizer updates',seed=background_seed,data_hash=sha256_file(ROOT/'experiments/runs/synthetic-split-manifest.json'),data_split='training and validation only',graph_summary='2 spins per word, lexical pairs and joint diagnostic window',limitations='Exploratory family-label-informed transfers on two synthetic seeds; no independent holdout, joint training, memory peak or physical hardware.')
    with mlflow.start_run(run_name=f'ising-family-mean-background{background_seed}-donor{donor_seed}') as run:
        log_contract(mlflow,contract,payload)
        mlflow.log_params({'background_seed':background_seed,'donor_seed':donor_seed,'endpoint':'fixed_lexical400','updates':400})
        mlflow.log_metrics({'weighted_anchor_tv':diag['anchor_total_variation'],'unweighted_anchor_tv':geometry['unweighted_anchor_tv'],'context_tv':diag['context_posterior_total_variation'],'sense_accuracy':diag['validation_macro_sense_accuracy'],'lexical_auc':auc,'validation_lexical_loss':val_loss,'train_lexical_loss':train_loss,'elapsed_seconds':elapsed})
        print(json.dumps({'run_id':run.info.run_id,'background':background_seed,'donor':donor_seed,'weighted_anchor_tv':diag['anchor_total_variation'],'context_tv':diag['context_posterior_total_variation'],'sense':diag['validation_macro_sense_accuracy'],'auc':auc,'checks':checks,'elapsed_seconds':elapsed}))
