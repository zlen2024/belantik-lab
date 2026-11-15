"""Append a linked provenance correction for historical F3 diagnostic hashes."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from scripts.run_ising_corrected_joint import snapshot_hash
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri

RUNS={17:'25f97015403c40498623fa2c8c3b8290',29:'232f088b9f2342c0ba2f03357d259bac',43:'91c8470602464d108dc5097820a5c85b'}

if __name__=='__main__':
    seed=int(sys.argv[1]); original=RUNS[seed]
    artifact=ROOT/'mlruns'/'3'/original/'artifacts'/'config.json'
    payload=json.loads(artifact.read_text(encoding='utf-8'))
    source=ROOT/'experiments'/'runs'/f'ising-attribution-original-seed{seed}.json'
    if sha256_file(source)!=payload['lexical_source_sha256']: raise ValueError('replay source hash mismatch')
    replay=json.loads(source.read_text(encoding='utf-8'))
    step=payload['training_metrics']['selection.joint_step']
    correction={'original_run_id':original,'original_artifact_sha256':sha256_file(artifact),'field_interpretation':'historical diagnosis checkpoint_sha256 contains replay source file hash, not selected checkpoint hash','lexical_replay_file_sha256':payload['lexical_source_sha256'],'lexical_snapshot_sha256':snapshot_hash(replay['snapshots']['lexical_selected_400']),'joint_selected_snapshot_sha256':snapshot_hash(payload['snapshots'][f'joint_selected_{step}']),'joint_fixed100_snapshot_sha256':snapshot_hash(payload['snapshots']['joint_100']),'hash_method':'SHA-256 of canonical UTF-8 JSON: sorted keys, separators comma/colon, Python numeric encoding'}
    mlflow.set_tracking_uri(tracking_uri()); mlflow.set_experiment('word-energy-exact-training')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='ising_corrected_f3_provenance_correction',energy_convention='beta E; F3 each spoke lambda/2',variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',sampler_schedule='read-only checkpoint hashing; no training',seed=seed,data_hash=sha256_file(ROOT/'experiments/runs/synthetic-split-manifest.json'),data_split='stored artifacts only',graph_summary='3 tokens, 2 spins, star',limitations='Corrects historical metadata field only; original artifacts preserved.')
    with mlflow.start_run(run_name=f'ising-corrected-f3-provenance-seed{seed}') as run:
        log_contract(mlflow,contract,correction)
        mlflow.set_tag('corrects_run_id',original)
        print(json.dumps({'correction_run_id':run.info.run_id,**correction}))
