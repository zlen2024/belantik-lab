# TWSI weak-preference pair pilot

21 November 2026. Protokol dan kod `4da0cde`; MLflow `dc05b25eaf7c4583b4478bc734a27ebe`. Data berasal daripada split eksploratori TWSI train yang sudah diperiksa, bukan official test atau dokumen yang terbukti berasingan. Tiga seed menggunakan checkpoint WikiText sedia ada dan lima epoch supervised. Ini latihan pada preference weak-label; bukan ujian model embedding sebagai pengganti umum, Gaussian, THRML atau chip.

| Model | Accuracy pasangan validation seed 17 / 29 / 43 | Semua pilihan betul dalam setiap reversal group seed 17 / 29 / 43 |
|---|---|---|
| Ising shared | 0.581 / 0.591 / 0.548 | 7 / 121, 4 / 121, 5 / 121 |
| Vector contextual | 0.571 / 0.565 / 0.568 | 14 / 121, 9 / 121, 10 / 121 |

Accuracy train Ising 0.853–0.860, vector 0.800–0.805; jurang besar terhadap validation menunjukkan fit pada data latihan tidak serta-merta pindah ke pasangan validation. Ia konsisten dengan overfitting atau perbezaan split, tetapi puncanya belum terasing. Accuracy validation pretrained sebelum pilot ialah 0.522–0.556 Ising dan 0.532–0.560 vector. Ini perbandingan pada split development yang sama, bukan keputusan bebas.

Buang identiti context pada checkpoint sama: pair accuracy Ising 0.578 / 0.581 / 0.563 dan vector 0.573 / 0.560 / 0.571; kedua-dua model mendapat 0/121 reversal group apabila hanya ranking tetap tinggal. Acak context sekali memberikan accuracy Ising 0.567 / 0.578 / 0.555 dan vector 0.575 / 0.551 / 0.559. Untuk reversal group, Ising 3 / 6 / 4 daripada 121, vector 13 / 11 / 11. Satu acakan bukan distribusi null; bilangan group berulang dan berkongsi target, jadi jangan anggap 121 pemerhatian bebas. Belum ada bukti kukuh bahawa model belajar menggunakan cue dengan betul, walaupun pair accuracy keseluruhan melebihi 0.5.

Validation terdiri daripada 186 target dan 1,362 pair group; hanya 121 group mempunyai kedua-dua arah. Group bidirectional melibatkan 50 target menurut critic split. Skor preference ialah weak supervision, konteks satu token mungkin tidak cukup, `the` adalah cue bagi 522/2,184 row, sumber dokumen dan near-duplicates belum dipastikan. Perturbasi context tanpa melatih semula tidak sama dengan baseline yang dilatih context-free. Tiada interval superiority: vector dan Ising mempunyai formula dan kos berbeza. Pilihan terakhir epoch tetap, tiada tuning berdasarkan result ini. Semakan independent critic direkod berasingan.
