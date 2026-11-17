# Analisis posthoc: kekerapan calon text8

17 November 2026. Selepas melihat keputusan pilot, critic GPT-6.1 Sol Ultra
mencadangkan analisis checkpoint sedia ada mengikut nisbah kekerapan konteks
positive kepada noise dalam train. Tiada training, tuning, held-out test atau
perubahan skor asal. Keputusan ini **deskriptif dan posthoc**, bukan ujian
statistik yang dibekukan sebelum pilot atau bukti penyebab jurang.

Gunakan 12 checkpoint epoch3, validasi/pasangan/noise/q yang sama, dan
corrected exact-ID tie rule. Asingkan hanya perbandingan positive/noise dengan
ID **berlainan**. Kelaskan setiap comparison menurut nisbah train count
`count(positive)/count(noise)`: `<0.5`, `[0.5,1)`, `=1`, `(1,2]`, `>2`.
Untuk setiap model/seed dan grup, simpan jumlah comparisons, AUC half-credit,
dan mean paired `ising-ranking − ising-shuffled`. Sertakan vector-ranking,
SGNS dan frequency-only. Pengiraan frequency-only bergantung pada count yang
mentakrifkan grup; skor dalam grup merupakan sanity/interpretation control,
bukan independent baseline evidence. Report grup bersaiz sifar sebagai null.

Data validation sampel repeated occurrences daripada satu stream tanpa
document boundaries; angka comparison besar tidak menjadikan observation
independent. Model seeds berkongsi data dan noise. Kategori kekerapan ini tidak
mengesahkan relatedness manusia atau makna bank, dan hubungan yang dijumpai
tidak membuktikan bahawa frekuensi **menyebabkan** jurang. Rekod sumber,
checkpoint/dataset hashes dan hasil dalam MLflow audit berasingan.
