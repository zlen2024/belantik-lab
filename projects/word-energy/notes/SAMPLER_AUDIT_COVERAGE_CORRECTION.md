# Pembetulan coverage — Gaussian joint sampler audit

**Tarikh:** 15 November 2026  
**Status:** selesai, direkod tanpa memadam artifact awal

## Apa yang berlaku

Audit sampler mula-mula dipecahkan kepada batch dua row kerana satu proses
panjang melepasi had masa interaktif. Ringkasan awal menyebut 12 row F/R itu
sebagai seluruh validation split. Semakan skrip `prepare_data()` dan corpus
manifest menunjukkan split sebenar mempunyai 18 row:

- row 0–5: keluarga F;
- row 6–11: keluarga R;
- row 12–17: keluarga M (distractor).

Jadi data row 0–11 tetap sah sebagai diagnostics per-window, tetapi claim
"split-wide" atau worst-case tidak sah sebelum row 12–17 diaudit.

## Semakan critic

Critic memeriksa `scripts/audit_gaussian_joint_sampler.py`, data preparation,
dan artifact audit. Dapatan penting:

1. Audit mengindeks `validation_windows` sahaja. Ia tidak score test window dan
   tidak memilih checkpoint baharu.
2. `_sense_map` menggunakan training anchors; family label digunakan hanya
   untuk laporan row, bukan sebagai input inference.
3. Checkpoint joint telah dipilih melalui validation lebih awal. Maka sampler
   audit mengesahkan ketepatan inference bagi checkpoint itu, bukan evaluasi
   generalisasi bebas atau pemilihan model baharu.
4. Coverage yang hilang ialah ralat pengukuran/protokol pelaporan, bukan bukti
   menyokong atau menolak hipotesis Word Energy.

## Pembaikan yang dibuat

Tanpa mengubah checkpoint, split, model, atau sampler schedule, jalankan row
12–13, 14–15 dan 16–17 bagi seeds 17, 29 dan 43. Semua batch menggunakan:

- four dispersed chains;
- burn-in 500 sweeps;
- 2,000 retained sweeps setiap chain;
- independence Metropolis-Hastings dan quadrature reference order 12;
- error gate `abs(error) <= max(0.02, 3 × MCSE)`, `R-hat < 1.05`,
  `ESS >= 400`, dan `MCSE <= 0.01`.

Batch digabung hanya selepas program mengesahkan setiap row 0–17 muncul tepat
sekali dan seed/burn-in/sample count semua batch sepadan. Artifact gabungan
dan run MLflow akhir:

| Seed | MLflow final run | Max R-hat | Min ESS | Max MCSE | Worst absolute error | Gate |
|---:|---|---:|---:|---:|---:|---|
| 17 | `bc5f465f9ca1418899799be6ef27dfdb` | 1.000668 | 5292.8 | 0.00656 | 0.01092 | pass |
| 29 | `1a666423fd884b95ba84d7b1fea8ecec` | 1.000829 | 4981.3 | 0.00671 | 0.01217 | pass |
| 43 | `7806a4c8ef4d4745a99cc4e115350b42` | 1.000718 | 5003.8 | 0.00703 | 0.01375 | pass |

## Provenance yang dibekukan bersama artifact gabungan

| Item | SHA-256 |
|---|---|
| Synthetic split manifest | `8b424e17a3100968b11a9b1a5e068c61d044ce66290f4e4269d72c014aa45d5d` |
| Sampler source | `9596a36c9f9b6a59ab950baabcfac070d4f8cc5f05a4997a1f42fe6ce0799a62` |
| Quadrature reference source | `18df358ad311de2e9aa8d8420def2d8cbbe943a2298ca2446b9e09889a05c5a9` |
| Audit script after coverage repair | `e03332dac1dc8fc3e95d7edff7eef88375e5a37bfe4c1290bb55d4e3f9f85406` |

| Seed | Selected checkpoint SHA-256 |
|---:|---|
| 17 | `472cd0673cd8c85e847688661e0800638d6afdae83a3c3e74276865d8d409a7d` |
| 29 | `ae4c68e9bb29ab7af432bae70a1e941726f8dbe099dc9aafee7194c3d320ce03` |
| 43 | `f93e0e18dc5a23d63b821bdf9e24d83c579c92b84cfd1317db7471f9de414c0f` |

## Batas kesimpulan

Ini ialah **classical MCMC** pada toy Gaussian D=1/M=2. Ia belum menguji
THRML, Torx, backend Extropic, atau physical thermodynamic chip. Ia juga tidak
mengubah keputusan sebelumnya bahawa Gaussian joint belum mengatasi frozen
lexical atau vector control pada contextual toy accuracy/retrieval.
