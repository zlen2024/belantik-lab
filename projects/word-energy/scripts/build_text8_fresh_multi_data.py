"""Create a reproducible fresh validation and fixed train permutations."""
from pathlib import Path
import json,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.train_text8_lexical_development import get_data
from scripts.audit_text8_canonical_evaluator import artifact_dir
from word_energy.text8_development import TEXT8_ZIP,verify_archive,region_tokens,make_pairs,noise_rows,sha_array
from word_energy.text8_lexical_models import pair_metrics
from word_energy.tracking import RunContract,log_contract,tracking_uri

BASE='6cf76591c3fa49c686a3d81d7f225607'; REGION=(17000000,17500000); PERMS=(11330,11331,11332)
def build():
    base,old=get_data(BASE); source=verify_archive()
    if source['content_sha256']!=old['source']['content_sha256']: raise ValueError('archive differs from frozen training data')
    vocab={w:i for i,w in enumerate(old['vocabulary'])}; words=region_tokens(TEXT8_ZIP,*REGION)
    ids=np.fromiter((vocab.get(w,-1) for w in words),dtype=np.int32,count=len(words))
    pairs,positions,stats=make_pairs(ids,5000,np.random.default_rng(11402))
    noise=noise_rows(base['noise_q'],pairs[:,1],20,np.random.default_rng(11403))
    arrays={'validation_pairs':pairs,'validation_positions':positions,'validation_noise':noise}
    original=base['train_pairs']
    diagnostics={}
    for seed in PERMS:
        order=np.random.default_rng(seed).permutation(len(original)).astype(np.int32)
        perm=original.copy(); perm[:,1]=original[order,1]
        if seed==11330 and (not np.array_equal(order,base['shuffle_order']) or not np.array_equal(perm,base['shuffled_train_pairs'])): raise ValueError('historical shuffle mismatch')
        if not np.array_equal(np.sort(order),np.arange(len(order))) or not np.array_equal(perm[:,0],original[:,0]) or not np.array_equal(np.sort(perm[:,1]),np.sort(original[:,1])): raise ValueError('invalid permutation')
        arrays[f'order_{seed}']=order; arrays[f'pairs_{seed}']=perm
        diagnostics[str(seed)]={'fixed_indices':int(np.sum(order==np.arange(len(order)))),'unchanged_context_ids':int(np.sum(perm[:,1]==original[:,1])),
          'unique_pairs':int(len(np.unique(perm,axis=0))),'self_pairs':int(np.sum(perm[:,0]==perm[:,1])),
          'positive_noise_collisions':[int(np.sum(perm[:,1,None]==e)) for e in base['train_noise']]}
    prior=.25*np.log(base['counts'].astype(float)); freq=pair_metrics(prior[pairs[:,1]],prior[noise],base['counts'][pairs[:,0]])
    metadata={'base_run_id':BASE,'base_array_hashes':old['hashes'],'source':source,'validation_region_bytes':list(REGION),
      'validation_seeds':[11402,11403],'permutation_seeds':list(PERMS),'validation_stats':stats,
      'validation_unique_position_pairs':int(len(np.unique(positions,axis=0))),
      'validation_noise_exact_collisions':int(np.sum(noise==pairs[:,1,None])),
      'permutation_diagnostics':diagnostics,'frequency_only':freq,'hashes':{k:sha_array(v) for k,v in arrays.items()},
      'limitations':'fresh byte block for development, no article boundaries and incomplete historical provenance; reserved test untouched'}
    return arrays,metadata

def load(run_id):
    c=MlflowClient(); run=c.get_run(run_id)
    if run.data.tags.get('model_family')!='text8_fresh_multi_data' or run.info.status!='FINISHED': raise ValueError('invalid data run')
    root=artifact_dir(c,run_id); m=json.loads((root/'config.json').read_text(encoding='utf-8'))
    with np.load(root/'dataset'/'samples.npz',allow_pickle=False) as a: values={k:a[k] for k in a.files}
    if {k:sha_array(v) for k,v in values.items()}!=m['hashes']: raise ValueError('fresh dataset mismatch')
    b,old=get_data(BASE)
    if old['hashes']!=m['base_array_hashes']: raise ValueError('base dataset mismatch')
    return b,values,m

if __name__=='__main__':
    mlflow.set_tracking_uri(tracking_uri()); mlflow.set_experiment('word-energy-text8-fresh-multi-shuffle')
    a,m=build(); contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='text8_fresh_multi_data',
      energy_convention='no model; frequency-only comparator',variable_encoding='word IDs',beta_temperature_convention='not applicable',
      sampler_schedule='fixed sampled validation and context permutations',seed=11402,data_hash=m['source']['content_sha256'],
      data_split='train 10-11M, old validation 16-16.5M, fresh validation 17-17.5M, reserved test untouched',graph_summary='not applicable',limitations=m['limitations'])
    with mlflow.start_run(run_name='text8-fresh-multi-data') as run:
        log_contract(mlflow,contract,m)
        tmp=ROOT/'.tmp';tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as d:
            p=Path(d)/'samples.npz';np.savez_compressed(p,**a);mlflow.log_artifact(str(p),'dataset')
        for p in (Path(__file__),ROOT/'word_energy/text8_development.py',ROOT/'notes/TEXT8_FRESH_MULTI_SHUFFLE_PROTOCOL.md'): mlflow.log_artifact(str(p),'source_snapshot')
        print(json.dumps({'run_id':run.info.run_id,'validation_stats':m['validation_stats'],'frequency_only':m['frequency_only'],'permutations':m['permutation_diagnostics']}))
