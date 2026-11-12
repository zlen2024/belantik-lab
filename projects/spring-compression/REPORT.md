# Keputusan: spring compression

Tarikh 2026-11-10. Eksperimen 1 dan 2 selesai; bukan ukuran hardware.

Semua invariant algebra lulus (float64, toleransi 1e-10).

## Autocorrelation mean mode

| Sistem | Integral exact continuous | Anggaran fixed window, 95% CI | Covariance error vs equilibrium sendiri |
|---|---:|---:|---:|
| clique | 1.0000 | 1.0108 [0.9831, 1.0385] | 0.0117 |
| star | 1.2500 | 1.2680 [1.2296, 1.3064] | 0.0158 |
| cycle | 1.0000 | 1.0041 [0.9882, 1.0201] | 0.0151 |

Covariance error ialah spectral norm selepas whitening. CI merentas seed; ACF menggunakan window 10, bukan integral infinity. Exact observation-grid/window tersedia dalam JSON.

## Ralat solution akhir terhadap clique asal

| Sistem | Uniform RMSE | Low-mode RMSE | High-mode RMSE |
|---|---:|---:|---:|
| clique | 1.289% | 2.883% | 2.883% |
| star | 1.242% | 2.777% | 2.777% |
| cycle | 1.375% | 36.105% | 21.429% |

RMSE empirikal merentas 10 seed; setiap seed menggabungkan 4 chain, horizon 5000 per chain. Ini bukan CI per run.

Jangan tafsir nilai RMSE star yang sedikit lebih rendah dalam run ini sebagai kelebihan. Expected RMSE analitik uniform ialah 1.266% (clique) dan 1.361% (star); variasi 10 seed boleh membalikkan ranking empirikal. Arah b berkongsi centered trajectory, maka error antaranya bukan replicate bebas.

## Tafsiran

- Star mengekalkan marginal visible tepat, tetapi tau mean mode meningkat 25% secara analitik dengan mobility=I.
- Cycle mempunyai bias low-mode 4/11 dan high-mode -4/19 pada amplitude mean; averaging tidak menghapuskannya.
- Mean dan covariance empirikal dibandingkan dengan equilibrium sistem sendiri. Cycle betul sebagai sampler bagi sistem baharu walaupun bias terhadap masalah asal.
- Relaksasi daripada initial state jauh diperiksa berasingan. Plot tidak digunakan untuk menentukan mixing time melalui crossing noisy.
- Sasaran 5% dinilai melalui expected RMSE stationary pada fixed grid; masa burn-in dan hardware programming tidak termasuk.
- Tempoh fizikal simulasi berbeza daripada wall-clock Python. Tiada bukti joule atau chip speedup.

## Artifak

- PROTOCOL.md: pilihan sebelum run.
- run_experiment.py: implementasi.
- results/results.json: konfigurasi, hashes, versions, algebra, 30 per-seed records, curves dan CI.
- results/experiment.png: plot.

Masa komputasi hingga JSON: 3.88 saat.

Sumber teori: [Aifer](https://threeplusone.com/pubs/Aifer2024a.pdf), [Kron](https://arxiv.org/abs/1102.2950), [sparsification](https://arxiv.org/abs/0803.0929).
Angka eksperimen ialah ukuran script tempatan; ramalan OU/star ialah derivasi untuk setup ini.
