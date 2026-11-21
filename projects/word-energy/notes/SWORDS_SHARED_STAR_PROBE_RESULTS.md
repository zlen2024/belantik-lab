# Sistem bersama pada SWORDS development: probe berpasangan

21 November 2026. Ini probe posthoc pada development yang sudah diperiksa berulang kali, bukan test bebas atau model baharu. Protokol dan code dibekukan pada commit `9ed4d92` sebelum pengiraan. Tanggal 22 November dalam protokol ialah kesilapan penulisan; tarikh run tempatan ialah 21 November. Ia menggunakan tepat subset 132 target informatif dan tiga checkpoint Ising sedia ada, dengan candidate-candidate serentak berada dalam satu star. Tiada training, Gaussian joint, THRML atau chip.

| Skor | Purata AUC 3 seed | Seed 17 / 29 / 43 |
|---|---:|---|
| Shared-star Ising | 0.57568 | 0.59237 / 0.55357 / 0.58109 |
| Shared-star dengan konteks uniform | 0.57718 | 0.60790 / 0.54861 / 0.57502 |
| Candidate-specific Ising sebelumnya | 0.58132 | 0.58059 / 0.58395 / 0.57943 |
| Vector target + context sebelumnya | 0.62272 | 0.61007 / 0.62734 / 0.63076 |

Purata perbezaan paired shared minus uniform ialah `-0.00150`, source-cluster bootstrap 95% `[-0.00733, 0.01424]`; shared minus candidate-specific `-0.00564` `[-0.04823, 0.00445]`; shared minus vector `-0.04705` `[-0.10758, -0.01306]`. Source-file ID hanyalah proxy, satu sumber dominan, dan tiada pembetulan untuk pemilihan berulang; jangan tafsir interval ini sebagai pengesahan superiority yang bebas. Penukaran kepada shared star mengubah 15.32% susunan pasangan candidate secara purata.

Apabila sumber dominan dibuang, tinggal 55 target: shared `0.52179`, uniform `0.51905`, candidate-specific `0.54197`, vector `0.58660`. Shared minus vector `-0.06481`, interval `[-0.11996, 0.00870]`. Keputusan ini tidak menunjukkan contextual gain yang boleh dipercayai atau kelebihan berbanding vector. Ia tidak menolak semua sistem bersama: checkpoint dilatih pada topology lain, coupling dinormalisasikan dengan bilangan semua candidate, dan data kecil serta banyak candidate terkeluar daripada vocabulary.

Semakan graph tambahan pada semua 219 manifest: degree maksimum setiap graph ialah `3+m+K`, dengan julat 7–55; 115 graph melebihi degree 16 dan 16 graph tepat degree 16. Angka 16 ialah had sasaran yang dipetik dalam pelan Z1, bukan pengesahan spesifikasi backend semasa atau kemampuan mapping kita. Graph yang melebihi had itu tidak boleh menjadi subgraph langsung di bawah andaian tersebut; degree lebih kecil juga tidak cukup untuk membuktikan mapping. Ini isu skala yang perlu diselesaikan, bukan bukti hardware gagal.

Per seed, seed 29 shared lebih rendah daripada Ising lama sebanyak `0.03038`; seed 17 dan 43 masing-masing lebih tinggi `0.01178` dan `0.00167`. Perubahan topology dan skala coupling berlaku serentak, jadi sebab perbezaan tidak terasing. Tiada tuntutan bahawa AUC adalah ukuran penuh kualiti embedding, kesesuaian perkataan umum, atau prestasi fizikal.

MLflow experiment `word-energy-natural-context-evaluation`, run `88afd4e4710a4da3b36080581905ff07`, mengandungi hashes, code, protocol, predictions, per-target metrics, dan perbandingan. Semakan critic bebas dan cadangan langkah berikut direkod secara berasingan.
