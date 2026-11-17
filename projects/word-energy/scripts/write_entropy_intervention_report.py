"""Write a report, including explicitly missing runs for an incomplete panel."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mlflow
from mlflow.tracking import MlflowClient
from scripts.run_ising_entropy_intervention_batch import PERMUTATIONS, ARMS
from scripts.run_ising_fresh_synthetic import SEEDS
from word_energy.tracking import tracking_uri

mlflow.set_tracking_uri(tracking_uri()); client=MlflowClient()
exp=client.get_experiment_by_name('word-energy-entropy-intervention-v1')
if exp is None: raise ValueError('experiment missing')
attempts={}
for run in client.search_runs([exp.experiment_id],max_results=1000):
    t=run.data.tags
    if not all(k in t for k in ('permutation_seed','seed','arm')): continue
    key=(int(t['permutation_seed']),int(t['seed']),t['arm'])
    path=ROOT/'mlruns'/exp.experiment_id/run.info.run_id/'artifacts'/'config.json'
    cfg=json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    attempts.setdefault(key,[]).append((run,cfg))
indexed={}; interrupted={}
for key,items in attempts.items():
    finished=[item for item in items if item[0].info.status=='FINISHED']
    other=[item for item in items if item[0].info.status!='FINISHED']
    if len(finished)>1 or len(other)>1: raise ValueError(f'duplicate attempts for {key}')
    if finished and other:
        if other[0][0].info.status!='KILLED' or finished[0][0].data.tags.get('retry_of_run_id')!=other[0][0].info.run_id:
            raise ValueError(f'unlinked retry for {key}')
        interrupted[key]=other[0][0].info.run_id
    indexed[key]=finished[0] if finished else other[0]
expected={(p,s,a) for p in PERMUTATIONS for s in SEEDS for a in ARMS}
if set(indexed)-expected: raise ValueError('unexpected run key')
complete=set(indexed)==expected and all(r.info.status=='FINISHED' and c is not None for r,c in indexed.values())
summary_id=sys.argv[1] if len(sys.argv)>1 else None
if complete and not summary_id: raise ValueError('run the full audit summary first')
summary=None
if summary_id:
    summary=json.loads((ROOT/'mlruns'/exp.experiment_id/summary_id/'artifacts'/'config.json').read_text(encoding='utf-8'))
finished=sum(r.info.status=='FINISHED' and c is not None for r,c in indexed.values())
lines=['# Keputusan galakan diversity sementara','',
       'Protokol dibekukan sebelum training: [protokol](ISING_ENTROPY_INTERVENTION_PROTOCOL.md). '
       'Galakan ialah perubahan loss latihan sementara pada corpus toy terpilih, bukan tenaga fizikal.','',
       f"Status panel: {'LENGKAP' if complete else 'TIDAK LENGKAP'}; {finished}/48 run siap. "
       + ('Kriteria dibaca daripada ringkasan yang telah diaudit.' if complete else 'Jangan menilai gate keseluruhan daripada hasil separa ini.'),'']
if interrupted:
    lines += ['Percubaan terputus dikekalkan dalam MLflow dan dipautkan kepada '
              f'run gantian: `{interrupted}`. Lihat [nota sambungan](ISING_ENTROPY_RESUMPTION_20261117.md).','']
if summary:
    lines += ['| Permutasi | Baseline | Early bank | Late bank | Early song |','|---:|---:|---:|---:|---:|']
    for perm,arms in summary['counts'].items():
        lines.append(f"| {perm} | {arms['baseline']}/4 | {arms['early-bank']}/4 | {arms['late-bank']}/4 | {arms['early-song']}/4 |")
    lines += ['',f"Kriteria beku: `{json.dumps(summary['frozen_criteria'],ensure_ascii=False)}`.",
              f"Median perubahan entropy bank lexical100: `{json.dumps(summary['median_paired_entropy_differences'])}` nats.",
              f'Ringkasan MLflow: `{summary_id}`.','']
lines += ['Satu run lulus jika sense >=10/12, AUC>=0.75 dan gain>=0.03 pada joint100. '
          'Entropy yang meningkat sahaja bukan bukti dua makna dipelajari. Ini eksperimen '
          'posthoc pada satu corpus menggunakan oracle semasa latihan; tiada test, '
          'teks sebenar atau hardware diukur.','',
          '## Rekod setiap run','',
          '| Permutasi | Seed | Arm | Run ID | Status | Sense | AUC | Gain | Lulus |',
          '|---:|---:|---|---|---|---:|---:|---:|---|']
for p in PERMUTATIONS:
    for s in SEEDS:
        for arm in ARMS:
            item=indexed.get((p,s,arm))
            if item is None:
                lines.append(f'| {p} | {s} | {arm} | — | belum dijalankan | — | — | — | — |'); continue
            run,cfg=item
            if run.info.status!='FINISHED' or cfg is None or 'fixed_joint100' not in cfg:
                lines.append(f'| {p} | {s} | {arm} | `{run.info.run_id}` | {run.info.status} | — | — | — | — |'); continue
            m=cfg['fixed_joint100']
            lines.append(f"| {p} | {s} | {arm} | `{run.info.run_id}` | selesai | {m['sense']:.3f} | {m['auc']:.3f} | {m['gain']:.6f} | {'ya' if m['pass'] else 'tidak'} |")
path=ROOT/'notes'/'ISING_ENTROPY_INTERVENTION_RESULTS.md'
path.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
print(path)
