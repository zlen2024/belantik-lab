# Pilot lexical text8: keputusan development

17 November 2026. Dua belas run klasik: empat arms × tiga model seed. Protokol dan kod dibekukan sebelum run: [protokol](TEXT8_LEXICAL_DEVELOPMENT_PROTOCOL.md), commit `dacab94`. Semua metrik ialah validation noise discrimination; ia bukan accuracy makna, benchmark document-independent, atau keputusan cip.

| Arm | AUC mean ± SD tiga seed | MRR@21 mean ± SD |
|---|---:|---:|
| ising-ranking | 0.6399 ± 0.0086 | 0.2745 ± 0.0084 |
| ising-shuffled | 0.6220 ± 0.0013 | 0.2424 ± 0.0017 |
| sgns | 0.6898 ± 0.0006 | 0.3280 ± 0.0069 |
| vector-ranking | 0.6786 ± 0.0051 | 0.3342 ± 0.0038 |
| Frequency-only, tanpa training | 0.6602 | 0.2658 |

Semua Ising ranking seeds melebihi Ising-shuffled pada AUC, tetapi ketiga-tiganya di bawah frequency-only dan kedua-dua vector arms pada AUC dalam pilot ini. Ising mean MRR lebih tinggi daripada frequency-only, jadi perbandingan bergantung kepada metrik. Ranking vector dan Ising berkongsi objective; SGNS menggunakan objective berbeza. Model juga berbeza parameter count dan simetri. Tiga seed berkongsi teks, sample dan noise; bukan tiga corpus bebas.

Lihat [semakan critic](TEXT8_LEXICAL_DEVELOPMENT_CRITIC.md) dan [analisis kekerapan posthoc](TEXT8_FREQUENCY_STRATA_POSTHOC_RESULTS.md).

Vocabulary 1,000 token berasal daripada train sahaja. Token coverage train 69.87% dan validation 66.19%; pasangan OOV tidak dinilai. Sample 50,000 train pairs dan 5,000 validation pairs mengandungi 21203 dan 3253 unique token pairs, serta 46066 dan 4916 unique occurrence-position pairs. Noise validation mempunyai 1122 exact-positive collisions daripada 100,000 draws; ties diberi half credit.

Metrik dalam laporan ini menggunakan [pembetulan exact-ID tie](TEXT8_IDENTICAL_CANDIDATE_METRIC_REPAIR.md): 10847 perbandingan merentas 48 checkpoint tersalah kira sebagai bukan tie disebabkan pembundaran batch. Run asal dan metrik asal dikekalkan; tiada training atau checkpoint diubah.

Text8 tidak mempunyai article boundaries; train/validation byte blocks berasingan tetapi dokumen sama boleh muncul pada kedua-duanya. Julat ini tidak ditemui dalam skrip lama yang diperiksa, tetapi provenance lama tidak lengkap. Reserved 18–20M tidak dinilai. Tiada human sense labels, WiC evaluation, THRML parity atau physical chip run.

MLflow experiment `word-energy-text8-lexical-development`: dataset run `6cf76591c3fa49c686a3d81d7f225607`, raw summary `790289d209054c48882b934809e60049`, corrected audit `e14d3b8e72e1467597b3826575be896f`. Sampel, noise, hashes dan checkpoint disimpan di MLflow tempatan; sumber code/protokol dan laporan di Git.

## Setiap run

| Arm | Seed | Run ID | AUC initial | AUC epoch3 | MRR@21 epoch3 | Masa (s) |
|---|---:|---|---:|---:|---:|---:|
| ising-ranking | 17 | `11a3f14f08b64f9e8dd97b2baad9e3f8` | 0.49307 | 0.64842 | 0.28082 | 16.3 |
| ising-ranking | 29 | `5f843719174f4056bf40162963973c66` | 0.49270 | 0.63126 | 0.26493 | 11.4 |
| ising-ranking | 43 | `a36c275079ae4d2d9fb380f86f4ab1c1` | 0.50271 | 0.64000 | 0.27762 | 14.1 |
| ising-shuffled | 17 | `05ce1730a46945168750b2132fc23c56` | 0.49307 | 0.62059 | 0.24183 | 11.4 |
| ising-shuffled | 29 | `a7e2e09b214046b69eb4468da197ff16` | 0.49270 | 0.62298 | 0.24103 | 15.0 |
| ising-shuffled | 43 | `207da45b8b0c4833a1a6be16b9fad514` | 0.50271 | 0.62253 | 0.24426 | 12.6 |
| sgns | 17 | `fb0b7a2707e946d9b207385082419a7d` | 0.49765 | 0.69004 | 0.32417 | 7.8 |
| sgns | 29 | `6e424af7bd4a4690805ad428fe0a9122` | 0.50087 | 0.68904 | 0.32378 | 6.5 |
| sgns | 43 | `f960769e5e874f3f980c3a909e757b17` | 0.49243 | 0.69019 | 0.33592 | 6.8 |
| vector-ranking | 17 | `f9602e9ffc8744d181e52a2743e7d4ae` | 0.49765 | 0.68400 | 0.33394 | 8.1 |
| vector-ranking | 29 | `b7a659d84c934a0d9a29e74eaca82488` | 0.50087 | 0.67796 | 0.33813 | 6.4 |
| vector-ranking | 43 | `f657d6f4d4e64e168b38a6b0573f36bc` | 0.49243 | 0.67387 | 0.33062 | 7.2 |
