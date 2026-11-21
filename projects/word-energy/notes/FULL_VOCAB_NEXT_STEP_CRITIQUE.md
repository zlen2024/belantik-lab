# Semakan langkah seterusnya — 21 November 2026

## Ising: objektif seluruh kamus

Hipotesis: latihan yang membandingkan perkataan tengah dengan semua 2,000 calon akan meningkatkan retrieval berbanding objektif margin sedia ada. Ini perlu diuji, termasuk kemungkinan bahawa ia tidak membantu.

Kod semasa menggunakan skor keserasian bersama offset `0.25 * log(count)` dan margin softplus terhadap calon noise. Pembetulan `log(noise_q)` pada readout dipilih untuk model margin selepas melihat validation. Cross-entropy seluruh kamus membandingkan semua calon secara terus dan menghasilkan sasaran berbeza. Oleh itu, gunakan skor latihan cross-entropy sendiri untuk ranking model baharu. Kekalkan pembetulan yang telah dibekukan untuk baseline margin. Perbezaan hasil merangkumi objektif dan peraturan readout yang berkaitan, jadi tidak boleh diatribusi kepada loss semata-mata. Laporkan juga kedua-dua peraturan readout pada kedua-dua model sebagai analisis diagnostik, tanpa memilih peraturan terbaik selepas melihat hasil.

Gunakan window, urutan, seed 17/29/43, parameter awal, optimizer dan bilangan update yang sama dengan baseline. Full softmax memproses lebih banyak calon setiap update, maka laporkan masa, memori dan bilangan calon yang diskor. Perbandingan update yang sama bukan perbandingan compute yang sama. Gunakan WikiText development sebagai panel utama dan EWT development sebagai semakan luaran yang telah digunakan sebelum ini. Kedua-duanya sudah berulang kali dilihat, jadi tidak boleh dianggap ujian akhir baharu. Kekalkan reserved test tertutup.

Gunakan article-level MRR dan perbezaan berpasangan merentas artikel sebagai ukuran utama, dengan purata tiga seed dan bootstrap artikel. Laporkan setiap seed, recall@10 dan baseline frequency serta vector. Tetapkan protokol dan kriteria sebelum latihan. Semak skor semua calon berbanding `score_windows` dan semak gradient/autodiff pada fixture kecil sebelum run. Semak nilai finite, checkpoint dan kos. Pengiraan ini ialah latihan klasik dengan penjumlahan tepat, maka keupayaan sampler dan chip mesti diuji berasingan.

## Gaussian: ujian dua petunjuk

Reka pasangan konteks yang jawapan sasaran memerlukan kedua-dua petunjuk dan ubah satu petunjuk pada satu masa. Sediakan kawalan lexical, satu petunjuk, additive dan vector yang menerima maklumat dan supervision yang sepadan. Periksa imbalance, duplikasi, kebocoran identiti token dan jalan pintas sebelum melatih. Pecahkan template dan perkataan mengikut kelompok untuk menilai generalisasi. Laporkan prestasi setiap keluarga dan seed. Dataset sintetik menguji mekanisme pada tugas yang dikawal, bukan bukti kefahaman bahasa semula jadi.

## Status

Ini semakan dan spesifikasi awal. Tiada eksperimen atau keputusan baharu dalam dokumen ini. Keputusan scaling terdahulu dan hadnya direkod dalam `FOUR_SPIN_SCALE_DECISION.md`.
