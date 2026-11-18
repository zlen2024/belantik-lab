# Text8: kawalan context shuffle dengan frequency offset

18 November 2026. Eksperimen posthoc development membandingkan pasangan train asal dengan satu permutasi context yang mengekalkan target dan histogram context. Kedua-duanya menggunakan frequency prior yang sama ketika train dan readout. [Protokol](TEXT8_FREQUENCY_SHUFFLE_PROTOCOL.md) dibekukan pada commit `fd63cac`.

## Hasil utama epoch3

| Model | Train | AUC purata | MRR@21 purata | AUC frequency hampir sama | AUC count sama |
|---|---|---:|---:|---:|---:|
| tied-frequency | original | 0.69803 | 0.33121 | 0.61397 | 0.59908 |
| tied-frequency | shuffled | 0.65519 | 0.26401 | 0.51849 | 0.48233 |
| role-frequency | original | 0.70038 | 0.34232 | 0.60817 | 0.56068 |
| role-frequency | shuffled | 0.65270 | 0.26171 | 0.51591 | 0.49462 |

Frequency hampir sama bermaksud ratio train count positive/noise antara 0.5 dan 2; 18,440 perbandingan distinct-ID. Count sama ada 217 perbandingan distinct-ID. Kedua-duanya metrik sekunder pada comparisons berkorelasi, bukan 18,440 atau 217 dokumen bebas. Frequency-only AUC bagi subgroup luas ialah 0.52362 dan exact-count 0.5.

| Model | Perbezaan AUC asal − acak: seed 17 / 29 / 43 | Purata | Kriteria utama | Perbezaan MRR purata |
|---|---|---:|---|---:|
| tied-frequency | +0.04665 / +0.03608 / +0.04579 | +0.04284 | lulus | +0.06720 |
| role-frequency | +0.04803 / +0.04620 / +0.04883 | +0.04769 | lulus | +0.08061 |

Kriteria deskriptif yang ditetapkan sebelum run: semua 3 kesan positif dan purata >= 0.005 AUC. Ini bukan p-value. Hasil selaras dengan susunan pasangan asal membantu discrimination terhadap noise pada validation ini. Namun, shuffled AUC (0.65519 tied dan 0.65270 role) lebih rendah daripada frequency-only (0.660195): sebahagian jurang boleh disebabkan oleh kemerosotan kawalan acak. Oleh itu, jurang penuh tidak mengukur hanya maklumat konteks yang berguna dan tidak memisahkan semantik daripada sifat statistik teks. Kesemua seed original turut mengatasi frequency-only secara berasingan.

## Setiap run

| Jenis | Model | Seed | Run ID | AUC | MRR | Masa (s) |
|---|---|---:|---|---:|---:|---:|
| original | tied-frequency | 17 | `8eab021c85ea4545bef2b804dec3b716` | 0.70210 | 0.33470 | 8.9 |
| original | tied-frequency | 29 | `52250706518949f0bd4d8886fdf5fde5` | 0.68881 | 0.32341 | 8.9 |
| original | tied-frequency | 43 | `d56b97169f46494baac3d86ef902217a` | 0.70319 | 0.33553 | 8.9 |
| shuffled | tied-frequency | 17 | `57dd54c2e7074eb5a116188c59d4b508` | 0.65545 | 0.26435 | 11.3 |
| shuffled | tied-frequency | 29 | `7c1cb04b59564464a8fbf4a0d9162178` | 0.65273 | 0.26221 | 9.0 |
| shuffled | tied-frequency | 43 | `38fba8d6fe0542158b5f06097107e044` | 0.65740 | 0.26548 | 12.3 |
| original | role-frequency | 17 | `5a141602f79445108dc872fb0d1c3880` | 0.70266 | 0.34245 | 8.6 |
| original | role-frequency | 29 | `1eb62d21661049979e488a452a76777b` | 0.69537 | 0.33815 | 10.2 |
| original | role-frequency | 43 | `c25f7f57ae3d4b7096de8e3468e30c01` | 0.70312 | 0.34638 | 10.3 |
| shuffled | role-frequency | 17 | `3f9bbfd521724c6ebb1165c742467acd` | 0.65463 | 0.26145 | 10.5 |
| shuffled | role-frequency | 29 | `6d59c915459d4840abc963baad30405c` | 0.64917 | 0.26064 | 12.0 |
| shuffled | role-frequency | 43 | `fe837ca5bfaf4258b0e9d846b1fceb9b` | 0.65429 | 0.26305 | 11.9 |

Enam run original menggunakan checkpoints factorial terdahulu; enam run shuffled ialah latihan baharu. Pasangan setiap seed mempunyai initialization, code model, data/noise, optimizer, epoch order dan frequency prior yang sama. Zero-coupling menghasilkan baseline frequency-only AUC 0.660195, MRR 0.265787273. MLflow menyimpan config, source, hash, semua checkpoint dan metrik setiap epoch.

## Pemeriksaan dan batas

- Permutasi mengubah hanya context IDs antara 50,000 train pairs. 0 indeks permutasi kekal di kedudukan asal, tetapi 1,052 baris masih mempunyai context ID yang sama. Pasangan unik 21,203 → 26,042; self-ID 949 → 1,035. Positive/noise collisions bagi tiga epoch: asal [2669, 2721, 2783], shuffled [2614, 2669, 2717]. Semua dikekalkan.
- Skor shuffled tidak wajib menjadi chance atau frequency-only: model boleh belajar marginal context yang disampel, pembetulan marginal dan hubungan rawak dalam satu permutasi. Perbezaan asal−acak mengukur kesan association pada prosedur ini, bukan pemisahan numerik bersih antara frequency dan semantik.
- Frequency-matched luas masih boleh mengandungi petunjuk frequency; exact count hanya 217 comparisons. Validation, contoh dan satu shuffle dikongsi oleh ketiga seed. Text8 tidak mempunyai article boundaries, dan positive/noise boleh berkongsi relevansi sebenar.
- Tiada labels sense, dataset bebas, reserved test, THRML parity atau cip fizikal diuji. Graph/frequency prior dibaca dan dilatih secara klasik; prior host belum dipetakan ke hardware.

MLflow experiment `word-energy-text8-frequency-shuffle-control`; summary `fe004a64eb5546af9e33ef34d703d5a2`. Enam original daripada `word-energy-text8-frequency-role-factorial`. Dataset run `6cf76591c3fa49c686a3d81d7f225607`. Checkpoints dan data disimpan di MLflow tempatan; nota dan kod disimpan dalam Git.

Semakan bebas dan tafsiran: [laporan critic](TEXT8_FREQUENCY_SHUFFLE_CRITIC.md).
