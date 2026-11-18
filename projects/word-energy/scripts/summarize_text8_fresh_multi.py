"""Audit all 36 endpoints on one fresh development validation set."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.build_text8_fresh_multi_data import load,BASE
from scripts.audit_text8_canonical_evaluator import artifact_dir
from word_energy.text8_canonical_eval import canonical_candidate_scores
from word_energy.text8_factorial_models import functions as ising_functions
from word_energy.text8_prior_vector import functions as vector_functions
from word_energy.text8_lexical_models import pair_metrics
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract,log_contract,tracking_uri,sha256_file

ARMS=('tied-frequency','role-frequency','vector-frequency');SEEDS=(17,29,43);SHUFFLES=(11330,11331,11332)

def collect(data_id):
    client=MlflowClient();base,fresh,meta=load(data_id);offset=.25*np.log(base['counts'].astype(float));pairs=fresh['validation_pairs'];noise=fresh['validation_noise'];counts=base['counts']
    names=['word-energy-text8-frequency-role-factorial','word-energy-text8-frequency-shuffle-control','word-energy-text8-fresh-multi-shuffle']
    histories=[]
    for name in names:
        exp=client.get_experiment_by_name(name);histories.append(client.search_runs([exp.experiment_id],max_results=1000))
    fresh_runs=[r for r in histories[2] if r.data.tags.get('arm') in ARMS]
    if len(fresh_runs)!=24:raise ValueError('expected exactly 24 fresh training runs')
    ratio=counts[pairs[:,1,None]]/counts[noise];mask=(pairs[:,1,None]!=noise)&(ratio>=.5)&(ratio<=2.)
    if not mask.any():raise ValueError('empty matched subset')
    freqpos=offset[pairs[:,1]];freqneg=offset[noise];freqwins=(freqpos[:,None]>freqneg)+.5*(freqpos[:,None]==freqneg)
    scorers={arm:(vector_functions(offset)[0] if arm=='vector-frequency' else ising_functions(arm,offset)[0]) for arm in ARMS}
    results={};initials={};snapshots={}
    for arm in ARMS:
        for seed in SEEDS:
            for sh in (0,*SHUFFLES):
                if arm=='vector-frequency' or sh in (11331,11332):
                    candidates=[r for r in fresh_runs if r.data.tags.get('arm')==arm and r.data.tags.get('seed')==str(seed) and r.data.tags.get('shuffle')==str(sh)]
                    new=True
                else:
                    records=histories[0] if sh==0 else histories[1]
                    candidates=[r for r in records if r.data.tags.get('arm')==arm and r.data.tags.get('seed')==str(seed)]
                    new=False
                if len(candidates)!=1 or candidates[0].info.status!='FINISHED':raise ValueError(f'missing/duplicate {arm} {seed} {sh}')
                run=candidates[0];root=artifact_dir(client,run.info.run_id);config=json.loads((root/'config.json').read_text(encoding='utf-8'))
                if not config['run_complete'] or config.get('offset_sha256')!=sha_array(offset):raise ValueError('status or offset')
                if (config['base_hashes'] if new else config['dataset_hashes'])!=meta['base_array_hashes']:raise ValueError('base mismatch')
                if new and (config['fresh_data_run']!=data_id or config['fresh_hashes']!=meta['hashes']):raise ValueError('fresh mismatch')
                expected=base['train_pairs'] if sh==0 else fresh[f'pairs_{sh}']
                if new and config['train_pair_hash']!=sha_array(expected):raise ValueError('train permutation mismatch')
                if not new:
                    if config['checkpoint_rule']!='fixed epoch3':raise ValueError('old checkpoint rule')
                    for rel in ('word_energy/text8_factorial_models.py','word_energy/text8_canonical_eval.py','word_energy/text8_lexical_models.py','word_energy/ising/exact_training.py','scripts/train_text8_lexical_development.py','word_energy/text8_development.py'):
                        p=ROOT/rel
                        if sha256_file(p)!=sha256_file(root/'source_snapshot'/p.name):raise ValueError('historical source mismatch')
                else:
                    hashes={p.name:sha256_file(p) for p in (root/'source_snapshot').iterdir() if p.is_file()}
                    if snapshots and hashes!=snapshots:raise ValueError('new source snapshots differ')
                    snapshots=hashes
                loaded={}
                for stage in ('initial','epoch1','epoch2','epoch3'):
                    with np.load(root/'checkpoints'/f'{stage}.npz',allow_pickle=False) as z:p={k:z[k] for k in z.files}
                    if {k:sha_array(v) for k,v in p.items()}!=config['checkpoint_hashes'][stage]:raise ValueError('checkpoint mismatch')
                    if stage in ('initial','epoch3'):loaded[stage]=p
                initials[(arm,seed,sh)]=loaded['initial']
                pos,neg=canonical_candidate_scores(loaded['epoch3'],scorers[arm],pairs,noise,symmetric=False)
                metrics=pair_metrics(pos,neg,counts[pairs[:,0]])
                if new:
                    for k in ('noise_pairwise_auc','noise_mrr_at_21'):
                        if not np.isclose(metrics[k],config['metrics']['epoch3'][k],atol=1e-14,rtol=0):raise ValueError('logged fresh metric mismatch')
                wins=(pos[:,None]>neg)+.5*(pos[:,None]==neg)
                results[f'{arm}:{seed}:{sh}']={'run_id':run.info.run_id,'reused':not new,'metrics':metrics,'matched_auc':float(wins[mask].mean()),'parameter_count':config['parameter_count']}
            initial=initials[(arm,seed,0)]
            for sh in SHUFFLES:
                other=initials[(arm,seed,sh)]
                if set(initial)!=set(other) or any(not np.array_equal(v,other[k]) for k,v in initial.items()):raise ValueError('paired initial mismatch')
    effects={}
    for arm in ARMS:
        matrix=[]
        for seed in SEEDS:
            original=results[f'{arm}:{seed}:0']['metrics']['noise_pairwise_auc']
            matrix.append([original-results[f'{arm}:{seed}:{sh}']['metrics']['noise_pairwise_auc'] for sh in SHUFFLES])
        arr=np.array(matrix);effects[arm]={'auc_original_minus_shuffle_seed_x_permutation':matrix,'all_permutation_mean':float(arr.mean()),'new_only_mean':float(arr[:,1:].mean()),'min':float(arr.min()),'max':float(arr.max()),'descriptive_gate':bool(np.all(arr>0) and arr.mean()>=.005)}
    return {'fresh_data_run':data_id,'base_data_run':BASE,'data_hashes':meta['hashes'],'base_hashes':meta['base_array_hashes'],'source_sha256':meta['source']['content_sha256'],
      'source_snapshot_hashes':snapshots,'frequency_only':meta['frequency_only'],'matched_count':int(mask.sum()),'matched_frequency_only_auc':float(freqwins[mask].mean()),'results':results,'effects':effects,
      'limitations':'posthoc development, one text8 region, incomplete historical provenance; shared seeds/permutations, no semantic or hardware claims'}

if __name__=='__main__':
    data_id=sys.argv[1];start=time.monotonic();mlflow.set_tracking_uri(tracking_uri());out=collect(data_id);out['audit_seconds']=time.monotonic()-start
    mlflow.set_experiment('word-energy-text8-fresh-multi-shuffle')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='text8_fresh_multi_summary',energy_convention='normalized F2 or dot score + host frequency prior',variable_encoding='spins or vectors',beta_temperature_convention='beta=1 Ising',sampler_schedule='canonical exact checkpoint scoring',seed=0,data_hash=out['source_sha256'],data_split='fresh dev validation 17-17.5M, reserved untouched',graph_summary='4 spin Ising or 2D vector',limitations=out['limitations'])
    with mlflow.start_run(run_name='text8-fresh-multi-summary') as run:
        log_contract(mlflow,contract,out)
        for arm,effect in out['effects'].items():mlflow.log_metric(f'{arm}.original_minus_shuffle_auc_mean',effect['all_permutation_mean']);mlflow.log_metric(f'{arm}.new_shuffle_only_mean',effect['new_only_mean'])
        mlflow.log_artifact(str(Path(__file__).resolve()),'source_snapshot')
        print(json.dumps({'run_id':run.info.run_id,'frequency_only':out['frequency_only'],'matched_count':out['matched_count'],'matched_frequency_only_auc':out['matched_frequency_only_auc'],'effects':out['effects'],'results':out['results']},indent=2))
