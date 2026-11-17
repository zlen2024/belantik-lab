# Analisis kekerapan calon: hasil posthoc

17 November 2026. Analisis ini menggunakan checkpoint epoch3 dan validation
pilot text8 yang telah dilihat, tanpa training tambahan atau test scoring.
[Protokol](TEXT8_FREQUENCY_STRATA_POSTHOC_PROTOCOL.md) menetapkan nisbah
kekerapan train dan memisahkan ID positive/noise yang sama. MLflow audit run
`be28eec775e340bd83e55db8f2f651ea` menyimpan setiap seed dan hash sumber.
GPT-6.1 Sol Ultra mengira semula setiap group untuk 12 checkpoint menggunakan
NumPy scorer berasingan; semua AUC dan counts sepadan. Kod critic dikira
read-only dan tidak disimpan; kod audit utama tersedia dalam Git/MLflow.

Setiap baris mengumpulkan perbandingan berdasarkan
`count_train(positive) / count_train(noise)`. Nilai model ialah mean AUC tiga
seed yang berkongsi teks/noise yang sama; jumlah comparisons juga sama bagi
semua model. Ini bukan bilangan dokumen bebas.

| Nisbah | Comparisons | Ising | Ising kocok | Vector ranking | SGNS | Frequency only |
|---|---:|---:|---:|---:|---:|---:|
| < 0.5 | 24,635 | 0.5297 | 0.1478 | 0.3710 | 0.2395 | 0.0 |
| [0.5, 1) | 8,676 | 0.5919 | 0.3929 | 0.5446 | 0.4933 | 0.0 |
| = 1 | 217 | 0.5576 | 0.4808 | 0.5591 | 0.5177 | 0.5 |
| (1, 2] | 9,547 | 0.6076 | 0.6383 | 0.6365 | 0.6538 | 1.0 |
| > 2 | 55,803 | 0.7047 | 0.8673 | 0.8465 | 0.9297 | 1.0 |

Sebanyak 98,878 distinct-ID comparisons dipecahkan ke lima kumpulan; 1,122
same-ID comparisons dikecualikan dan telah ditangani sebagai ties dalam
[laporan keseluruhan](TEXT8_LEXICAL_DEVELOPMENT_RESULTS.md). Dalam kelompok
positive lebih jarang, Ising yang dilatih pada context benar mengatasi
kawalan Ising kocok; dalam kelompok positive jauh lebih kerap, urutannya
terbalik. Jurang mean paired Ising minus kocok ialah +0.3819 untuk `<0.5`
dan −0.1626 untuk `>2`.
Mean `<0.5` tidak bermaksud semua seed melepasi 0.5: seed 29 masih di bawah
chance. Kumpulan equal-frequency pula hanya mempunyai 217 comparisons.

Frequency-only mesti memberi 0/0.5/1 mengikut kategori di atas kerana formula
dan kategori kedua-duanya dibina daripada count yang sama; ia bukan dapatan
bebas. Kumpulan `>2` merangkumi lebih separuh comparisons, jadi ia mempengaruhi
aggregate AUC. Pemerhatian ini konsisten dengan model menggunakan statistik
frekuensi secara berbeza, tetapi tidak membuktikan frequency sebagai punca
atau bahawa konteks jarang mewakili similarity semantik yang betul. Pasangan
dan seed berkongsi data; text8 tiada article boundaries; coverage validation
sekitar 66.19%. Tiada human-sense evaluation, THRML atau physical chip run.
