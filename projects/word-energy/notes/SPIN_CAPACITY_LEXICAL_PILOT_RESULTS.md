# Pilot kapasiti: 4 dan 6 spin dengan W yang dilatih

22 November 2026. Protokol/code `c61cdc3`; 12 run siap dalam MLflow experiment `word-energy-spin-capacity-lexical`. Ujian ini memakai 10,000 window WikiText train, 2,000 window development yang sudah pernah dilihat, lima noise per contoh dan tiga seed. Tiada model dilatih pada test rasmi. Ini ukuran membezakan perkataan observed daripada noise, bukan accuracy sense, lexical substitution atau prestasi embedding menyeluruh.

| Model | Bilangan keadaan unary | AUC pada epoch 3, seed 17 / 29 / 43 | Purata |
|---|---:|---|---:|
| Ising 4 spin | 16 | .64800 / .64635 / .65475 | .64970 |
| Ising 6 spin | 64 | .65380 / .63915 / .63985 | .64427 |
| Vector 10 dimensi | — | .65000 / .65615 / .65425 | .65347 |
| Vector 21 dimensi | — | .64405 / .64190 / .65510 | .64702 |
| Prior kekerapan sahaja | — | sama untuk semua seed | .66370 |

Tiga epoch dan checkpoint akhir ditetapkan sebelum run; jangan pilih epoch terbaik setelah melihat data dan panggil ia keputusan confirmatory. Epoch pertama Ising 4 memberikan .66475/.66625/.66770; 6 memberikan .66445/.66200/.66775. Jatuh selepas itu konsisten dengan masalah generalization/optimization pada konfigurasi kecil ini, tetapi belum menetapkan sebabnya. Noise samples boleh mengandungi alternatif yang sesuai. Ini **bukan** perbandingan terus dengan AUC .7 lama yang menggunakan tugasan/konfigurasi lain. Seed dan jangka masa kecil tidak menyokong kesimpulan universal bahawa menambah spin memudaratkan.

W yang pengguna cadangkan memang dilatih: jumlah coupling diagonal bermula .800 untuk setiap run. Selepas tiga epoch, Ising 4 menjadi 1.319/1.316/1.311 dan Ising 6 menjadi 1.290/1.281/1.285. Setiap coordinate mempunyai parameter sendiri, positif dan dibatasi; hanya had maksimum jumlah coupling ditetapkan sama pada 1.5. Ia analogi kepada kekuatan spring, bukan ukuran kekuatan fizikal. Ini membuktikan parameter berubah, bukan membuktikan perubahan itu mengajar makna; semua run menghampiri had 1.5 yang dipilih, jadi kemungkinan batas/tepu wajar diperiksa sebelum scale.

Menambah spin juga mengubah bilangan parameter unary (10→21 per word), sambungan dan kos; `2^d` ialah bilangan konfigurasi, bukan bilangan ayat yang model boleh ingat atau bilangan makna yang benar-benar dipelajari. Ising 6 seed 17 mengatasi 4 seed 17, tetapi 6 kalah pada dua seed lain dan purata lebih rendah. Tiada peningkatan yang stabil atau kemenangan terhadap frequency baseline pada epoch akhir. Vector berkongsi data, objective dan budget update tetapi perbezaan bentuk dan kos pengiraan perlu diingat.

Semua pengiraan ialah exact classical CPU reference; tiada THRML parity, Gaussian natural-text comparator atau chip diuji. Ujian matematik kecil lulus sebelum eksperimen. Critic bebas 6.2 Sol medium memeriksa code, W, metrik dan hipotesis dalam nota berasingan.

Metric bernama `noise_mrr_at_21` dalam run sebenarnya menggunakan 5 noise (`MRR@6`), kerana helper mewarisi nama lama; AUC di atas tidak terjejas. JAX scorer juga memerlukan ID divalidasi oleh caller dan tidak menolak semua ID tidak sah sendiri; critic mengesahkan data panel sah, tetapi invalid-ID unit test yang dijanjikan dalam protokol tidak dilakukan. Butiran dan semakan bebas ada dalam [kritik](SPIN_CAPACITY_LEXICAL_PILOT_CRITIC.md).
