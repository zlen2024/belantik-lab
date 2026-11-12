# Scaling dan calibration error

2026-11-10. Synthetic-device study dengan exact linear algebra; bukan hardware measurement.

## Setup

n=16,64,256; clique weight 1/n; star weight 1; g=1. A=2I-11^T/n, condition number 2. Normalisasi ini berbeza daripada eksperimen asal n=4 weight1.
Bound delta merujuk kepada Uniform[-delta,+delta] relative component errors, bukan standard deviation. 20 seed per condition; grounding paired. Force directions random tetap untuk semua devices.
Worst-direction relative A-norm solution bias dikira untuk semua b melalui eigenvalues. Ia sama dengan whitened spectral covariance error.

## Hasil utama: coupling dan grounding kedua-duanya tidak tepat

| n | Bound | Sistem | Median worst bias | Median random-force bias | Devices lulus <=5% |
|---:|---:|---|---:|---:|---:|
| 16 | 5% | clique | 2.91% | 1.65% | 20/20 |
| 16 | 5% | star | 4.09% | 2.00% | 19/20 |
| 16 | 10% | clique | 5.87% | 3.29% | 3/20 |
| 16 | 10% | star | 8.06% | 3.96% | 0/20 |
| 64 | 5% | clique | 2.75% | 1.48% | 20/20 |
| 64 | 5% | star | 4.59% | 1.96% | 19/20 |
| 64 | 10% | clique | 5.61% | 2.95% | 0/20 |
| 64 | 10% | star | 9.61% | 3.93% | 0/20 |
| 256 | 5% | clique | 2.71% | 1.43% | 20/20 |
| 256 | 5% | star | 4.81% | 2.05% | 17/20 |
| 256 | 10% | clique | 5.56% | 2.87% | 0/20 |
| 256 | 10% | star | 10.10% | 4.09% | 0/20 |

Pass rate ialah antara synthetic devices sahaja; Wilson 95% intervals ada dalam JSON. Zero failures dalam 20 devices tidak menjamin tiada hardware failures.
Mean 95% CI dan empirical 5/95 percentiles disimpan berasingan dalam JSON. Semua delta/scopes yang berkongsi calibration draws berkorelasi.

## Sensitivity mengikut komponen, n=256 dan bound 10%

| Scope | Clique median worst bias | Star median worst bias |
|---|---:|---:|
| coupling_only | 0.59% | 5.23% |
| grounding_only | 5.52% | 5.52% |
| both | 5.56% | 10.10% |

Coupling-only mengekalkan uniform solution tepat kerana semua grounding kekal 1. Ujian uniform sahaja tidak mengesan coupling error ini.
## Common-mode dan auxiliary parasitic grounding

Jika semua coupling dan grounding tersasar -10% bersama, A_actual=.9 A: relative solution bias 11.111% untuk kedua-dua architecture, semua n. Ini tidak hilang melalui iid averaging atau thermal averaging.
Auxiliary grounding asal sifar: ralat multiplicative sahaja tidak mengujinya. Ujian tambahan additive eta menghasilkan uniform bias eta/(n+2*eta). Formula awal diperbetulkan selepas assertion Schur-complement gagal; perubahan dicatat dalam protokol.

| n | Auxiliary parasitic ground eta | Worst bias |
|---:|---:|---:|
| 16 | 0.01 | 0.062% |
| 16 | 0.1 | 0.617% |
| 64 | 0.01 | 0.016% |
| 64 | 0.1 | 0.156% |
| 256 | 0.01 | 0.004% |
| 256 | 0.1 | 0.039% |

## Thermal sampling: ramalan analitik, bukan trajectory measurements

Primary calibration study mengasingkan bias menggunakan exact solves. Thermal noise diperkirakan secara berasingan menggunakan exact covariance continuous-time stationary average, 4 chains.
Unit-norm force ditetapkan; semakin banyak coordinates, semakin besar jumlah noise vector relatif kepada signal ini. Ini bergantung pada force normalization dan bukan universal scaling bagi setiap workload.

| n | Sistem | Expected uniform RMSE T=5000 | Expected contrast RMSE T=5000 |
|---:|---|---:|---:|
| 16 | clique_ideal | 2.92% | 4.12% |
| 16 | star_ideal_r1 | 2.93% | 4.14% |
| 64 | clique_ideal | 5.70% | 8.06% |
| 64 | star_ideal_r1 | 5.70% | 8.06% |
| 256 | clique_ideal | 11.34% | 16.03% |
| 256 | star_ideal_r1 | 11.34% | 16.03% |

Untuk noisy seed0 bound10%, JSON memisahkan bias_squared, sampling_variance, dan total_expected_rmse pada setiap horizon. Seed0 ialah ilustrasi ditetapkan, bukan typical-device estimate.
Bias calibration tidak turun dengan T. Formula error decomposition ialah exact expectation: total MSE = squared bias + variance, dalam A-norm.
Ideal star mean-mode tau=1+1/(n*r): penalti mengecil dengan n untuk family ini, tetapi auxiliary degree=n dan fastest mode meningkat. Ini tidak membuktikan scalability hardware.

## Semakan dan batas

18 independent direct-solve worst-direction checks lulus; star full-vs-Schur solves, ideal invariants, perturbation bound dan coupling-only uniform invariants lulus.
Semua bounded calibration cases memenuhi worst bias <=delta/(1-delta). Formula ini membekalkan guarantee konservatif; pass fraction hasil synthetic model memberi bukti lebih terhad.
Dense eigensolver digunakan untuk audit, bukan comparison runtime. Digital clique baseline mempunyai structured O(n) solve. Tiada training, optimization, large thermal trajectories, chip energy, maximum-degree routing atau circuit validation.
Wall-clock sebelum plot: 5.61 saat.

## Fail

- SCALING_PROTOCOL.md: protokol sebelum run.
- run_scaling.py: kod.
- scaling_results/results.json: metadata/hash, 2160 device-condition records, summary, controls dan exact sampling predictions.
- scaling_results/scaling.png: figure.

Formula dan hasil ialah derivasi serta numerical experiments tempatan. Calibration distribution sebenar belum diukur; tiada dakwaan toleransi hardware.
