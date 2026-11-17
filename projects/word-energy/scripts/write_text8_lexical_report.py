"""Write the text8 development report from the audited MLflow summary."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
summary_id=sys.argv[1]
audit_id=sys.argv[2]
path=ROOT/'mlruns'/'12'/summary_id/'artifacts'/'config.json'
data=json.loads(path.read_text(encoding='utf-8'))
corrected=json.loads((ROOT/'mlruns'/'12'/audit_id/'artifacts'/'config.json').read_text(encoding='utf-8'))
if len(data['runs'])!=12: raise ValueError('full panel required')
if corrected['raw_summary_run_id']!=summary_id or len(corrected['runs'])!=12: raise ValueError('correction provenance mismatch')
freq=data['frequency_only_control']
lines=['# Pilot lexical text8: keputusan development','',
       '17 November 2026. Dua belas run klasik: empat arms × tiga model seed. '
       'Protokol dan kod dibekukan sebelum run: [protokol](TEXT8_LEXICAL_DEVELOPMENT_PROTOCOL.md), '
       'commit `dacab94`. Semua metrik ialah validation noise discrimination; '
       'ia bukan accuracy makna, benchmark document-independent, atau keputusan cip.', '',
       '| Arm | AUC mean ± SD tiga seed | MRR@21 mean ± SD |',
       '|---|---:|---:|']
for arm,a in corrected['aggregate'].items():
    lines.append(f"| {arm} | {a['noise_auc_mean']:.4f} ± {a['noise_auc_seed_sd']:.4f} | {a['noise_mrr_mean']:.4f} ± {a['noise_mrr_seed_sd']:.4f} |")
lines += [f"| Frequency-only, tanpa training | {freq['noise_pairwise_auc']:.4f} | {freq['noise_mrr_at_21']:.4f} |",'',
          'Semua Ising ranking seeds melebihi Ising-shuffled pada AUC, tetapi '
          'ketiga-tiganya di bawah frequency-only dan kedua-dua vector arms '
          'pada AUC dalam pilot ini. Ising mean MRR lebih tinggi daripada '
          'frequency-only, jadi perbandingan bergantung kepada metrik. '
          'Ranking vector dan Ising berkongsi objective; SGNS menggunakan '
          'objective berbeza. Model juga berbeza parameter count dan simetri. '
          'Tiga seed berkongsi teks, sample dan noise; bukan tiga corpus bebas.', '',
          'Lihat [semakan critic](TEXT8_LEXICAL_DEVELOPMENT_CRITIC.md) dan '
          '[analisis kekerapan posthoc](TEXT8_FREQUENCY_STRATA_POSTHOC_RESULTS.md).', '',
          'Vocabulary 1,000 token berasal daripada train sahaja. Token coverage '
          f"train {100*data['train_stats']['token_coverage']:.2f}% dan validation "
          f"{100*data['validation_stats']['token_coverage']:.2f}%; pasangan OOV "
          'tidak dinilai. Sample 50,000 train pairs dan 5,000 validation pairs '
          f"mengandungi {data['train_unique_pairs']} dan {data['validation_unique_pairs']} "
          'unique token pairs, serta '
          f"{data['train_unique_occurrence_pairs']} dan {data['validation_unique_occurrence_pairs']} "
          'unique occurrence-position pairs. '
          f"Noise validation mempunyai {data['validation_exact_noise_collisions']} "
          'exact-positive collisions daripada 100,000 draws; ties diberi half credit.', '',
          'Metrik dalam laporan ini menggunakan [pembetulan exact-ID tie](TEXT8_IDENTICAL_CANDIDATE_METRIC_REPAIR.md): '
          f"{corrected['total_false_nonties']} perbandingan merentas 48 checkpoint tersalah kira sebagai bukan tie "
          'disebabkan pembundaran batch. Run asal dan metrik asal dikekalkan; '
          'tiada training atau checkpoint diubah.', '',
          'Text8 tidak mempunyai article boundaries; train/validation byte blocks '
          'berasingan tetapi dokumen sama boleh muncul pada kedua-duanya. Julat ini '
          'tidak ditemui dalam skrip lama yang diperiksa, tetapi provenance lama '
          'tidak lengkap. Reserved 18–20M tidak dinilai. Tiada human sense labels, '
          'WiC evaluation, THRML parity atau physical chip run.', '',
          f"MLflow experiment `word-energy-text8-lexical-development`: dataset run `{data['dataset_run_id']}`, "
          f"raw summary `{summary_id}`, corrected audit `{audit_id}`. Sampel, noise, hashes dan checkpoint "
          'disimpan di MLflow tempatan; sumber code/protokol dan laporan di Git.', '',
          '## Setiap run','',
          '| Arm | Seed | Run ID | AUC initial | AUC epoch3 | MRR@21 epoch3 | Masa (s) |',
          '|---|---:|---|---:|---:|---:|---:|']
for key,row in sorted(data['runs'].items()):
    arm,seed=key.rsplit(':',1); m=corrected['runs'][key]['stages']
    lines.append(f"| {arm} | {seed} | `{row['run_id']}` | {m['initial']['corrected']['noise_pairwise_auc']:.5f} | {m['epoch3']['corrected']['noise_pairwise_auc']:.5f} | {m['epoch3']['corrected']['noise_mrr_at_21']:.5f} | {row['elapsed_seconds']:.1f} |")
output=ROOT/'notes'/'TEXT8_LEXICAL_DEVELOPMENT_RESULTS.md'
output.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
print(output)
