"""Write a Malay ledger from an audited MLflow panel summary."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
run_id = sys.argv[1]
experiment_id = sys.argv[2]
summary = json.loads((ROOT/'mlruns'/experiment_id/run_id/'artifacts'/'config.json').read_text(encoding='utf-8'))
if len(summary['runs']) != 32 or len(summary['permutation_pass_counts']) != 8:
    raise ValueError('expected complete 32-run panel')
lines = [
    '# Corpus 227: eksperimen lapan permutasi', '',
    'Susulan eksploratori 16 November 2026. Protokol dan kod latihan dibekukan '
    'sebelum panel (commit `49bcd6b`): [protokol](ISING_CORPUS227_PERMUTATION_PANEL_PROTOCOL.md). '
    'Corpus ini dipilih selepas keputusan asal dilihat; keputusan di sini tidak '
    'mengubah kegagalan gate enam corpus.', '',
    'Setiap baris ialah satu susunan berbeza dengan empat model seed. Lulus '
    'bermaksud sense >=10/12, lexical AUC >=0.75 dan gain >=0.03 pada joint100. '
    'Bilangan dalam kurungan ialah seed yang lulus, bukan dataset bebas.', '',
    '| Seed permutasi | Bilangan lulus daripada 4 |', '|---:|---:|',
]
for perm,count in summary['permutation_pass_counts'].items():
    lines.append(f'| {perm} | {count}/4 |')
joint_total=sum(row['joint100']['pass'] for row in summary['runs'].values())
lex_total=sum(row['lexical400']['pass'] for row in summary['runs'].values())
lines += [
    '', f'Jumlah run lulus ialah {joint_total}/32 pada joint100 dan {lex_total}/32 '
    'pada lexical400. Ini adalah ukuran deskriptif bagi lapan susunan pada '
    'satu corpus, bukan 32 replikasi dataset. Susunan 7019 juga 4/4 seperti '
    'RNG6061, tetapi 7043 hanya 0/4: hasil bergantung pada susunan dan seed.', '',
    'Kawalan corpus 227 terdahulu: spectral 4/4, random 2/4, '
    'permutasi RNG6061 4/4. Empat model seed ialah 251, 257, 263 dan 269; '
    'susunan lama telah dipilih untuk susulan kerana 4/4, jadi bukan sampel '
    'tanpa pemilihan bersama lapan susunan baru. Kejayaan beberapa susunan '
    'melemahkan idea bahawa hanya RNG6061 boleh berjaya pada corpus ini; '
    'ia tidak membuktikan punca atau manfaat susunan spectral secara umum.', '',
    'Semakan posthoc critic GPT-6.1 Sol Ultra membandingkan 7019 (4/4) '
    'dengan 7043 (0/4): bagi 7043, dua keluarga anchor pada akhirnya '
    'hampir menggunakan keadaan yang sama (purata anchor TV 0.00550), '
    'berbanding 0.99737 untuk 7019. Kedua-duanya mempunyai AUC 1.0. '
    'Pada awal latihan, centroid F–R 7043 sebenarnya lebih jauh; ini '
    'bermakna jarak awal semata-mata tidak menjelaskan keputusan. '
    'Ini pemerhatian terhadap dua susunan terpilih, bukan bukti punca. '
    'Lihat [audit dan ukuran semua permutasi](ISING_CORPUS227_PERMUTATION_CRITIC.md).', '',
    'Data latihan, oracle bagi negatives/corruptions, corrected F3, 400 '
    'lexical updates, handoff dan 100 joint updates kekal. Semua run klasik '
    'exact enumeration; tiada test, teks sebenar, THRML atau hardware diukur. '
    'Lapan permutasi menyiasat satu corpus terpilih dan tidak membuktikan '
    'mekanisme sebab-akibat atau generalisasi ke corpus lain.', '',
    f'MLflow: `word-energy-corpus227-permutation-panel-v1`, summary run `{run_id}`. '
    'Artifak setiap run menyimpan konfigurasi, hash, titik semakan dan metrik.', '',
    '## Rekod setiap run', '',
    '| Permutasi | Model seed | Run ID | Lexical pass | Sense joint | AUC | Gain | Joint pass |',
    '|---:|---:|---|---|---:|---:|---:|---|',
]
for key, row in summary['runs'].items():
    p,s=key.split(':'); m=row['joint100']; lex=row['lexical400']
    lines.append(f"| {p} | {s} | `{row['run_id']}` | {'ya' if lex['pass'] else 'tidak'} | {m['sense']:.3f} | {m['auc']:.3f} | {m['gain']:.6f} | {'ya' if m['pass'] else 'tidak'} |")
path=ROOT/'notes'/'ISING_CORPUS227_PERMUTATION_PANEL_RESULTS.md'
path.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
print(path)
