# 6 berbanding 8 spin: heldout artikel sumber train

22 November 2026. Model, data dan metrik dibekukan di `16d0dfa`; pemuatan checkpoint dibetulkan di `58f4175` selepas cubaan evaluation pertama gagal sebelum skor dikeluarkan. Lihat [rekod pembaikan](SPIN8_EVALUATION_REPAIR.md) dan MLflow `6435475dc7d543b3aa994cd84e291caf`. Enam model telah dilatih sebelum penilaian pada 60 artikel yang dikeluarkan daripada latihan, pembinaan vocabulary, count dan noise distribution run ini. Dataset `3387fc4fb99646f1b7aacd518f39bf10`, evaluation `c4eb4b4e2e4b4bb09c0455774b40b594`. Artikel berasal daripada WikiText train yang pernah digunakan dalam kajian dahulu; ini bukan benchmark luaran yang sepenuhnya belum pernah dilihat.

| Model | Document-macro AUC seed 17 / 29 / 43 | Purata |
|---|---|---:|
| Ising 6 spin | .63629 / .64695 / .65060 | .64461 |
| Ising 8 spin | .64228 / .64348 / .64600 | .64392 |
| Prior kekerapan | sama untuk semua seed | .65459 |

Purata perbezaan paired document-macro 8−6 ialah −.00069, interval bootstrap artikel 95% `[-.00933,.00834]`, bersyarat pada tiga seed. Perbezaan setiap seed +.00599/−.00347/−.00460. Ising8−frequency purata −.01067 `[-.01981,−.00150]`. **Gate yang ditetapkan sebelum run tidak lulus:** 8 spin tidak mengatasi 6 spin untuk semua seed dan kedua-duanya tidak mengatasi frequency. Interval yang merangkumi sifar bukan bukti kedua-duanya setara.

Pair AUC mentah Ising6 .64865/.65475/.65725, Ising8 .64715/.64745/.65245 dan frequency .65610; ini contoh observed versus lima noise, bukan ketepatan pemilihan makna. Validation merangkumi 2,000 contoh dari 60 artikel; vocabulary meliputi 72.2% token heldout. Lima noise boleh termasuk alternatif valid; 85 daripada 10,000 comparison ialah collision exact dan dikira separuh. Penilaian memerlukan satu konteks token dan tidak menguji idea `bank` dua makna.

Enam run menggunakan `W` relatif yang dilatih dengan jumlah .8; 8 spin mempunyai 256 konfigurasi unary dan 36 parameter per perkataan berbanding 64 dan 21 untuk 6 spin. Lebih banyak konfigurasi tidak secara automatik bermakna lebih banyak ayat/makna yang dipelajari. Penambahan spin juga mengubah graph, parameter dan kos, jadi keputusan ini tidak mengasingkan kesan satu pemboleh ubah semata-mata. Tiada training atau skor pada official validation/test, Gaussian, THRML atau chip. Critic bebas diminta menyemak seluruh pipeline dan hipotesis.

Untuk 8-spin, setiap W bermula .100; nilai akhir merangkumi .08370–.11323 merentas tiga seed, dengan jumlah tetap .8. Tempoh setiap run pada CPU kira-kira 14.8–15.6 saat dan peak working set yang diukur 482–486 MiB, berbanding 8.9–10.8 saat dan 460–464 MiB bagi 6-spin. Ini pengukuran proses CPU untuk konfigurasi kecil ini sahaja, bukan kecekapan atau penggunaan tenaga chip.
