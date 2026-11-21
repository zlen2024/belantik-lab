# Keputusan full-softmax prior ablation — 21 November 2026

## Soalan dan kaedah

Selepas pilot suku log-count merosot, kami uji sama ada menggunakan `log(count)` semasa latihan full-softmax menaikkan kedudukan perkataan sebenar. Perubahan tunggal yang dirancang ialah offset kekerapan. Model joint4, 200,000 window, tiga pass, optimizer, susunan dan seed 17/29/43 dikekalkan. Penilaian menggunakan skor yang sama seperti latihan. Rujukan perbandingan berasal daripada run sebelumnya, dengan ranking dipadankan pada window dan dokumen sama. Semua data penilaian sudah pernah diperiksa dan reserved test kekal tertutup.

## Hasil

MRR lebih tinggi bermaksud perkataan sebenar cenderung mendapat rank lebih baik. Nilai berikut ialah purata per dokumen, kemudian purata tiga seed.

| Panel | Full-softmax awal | Awal dengan readout diperbetulkan | Full-softmax log-count baharu | Margin | Frequency | Vector10 |
|---|---:|---:|---:|---:|---:|---:|
| WikiText, 60 artikel | 0.16862 | 0.18376 | 0.19800 | 0.19448 | 0.18264 | 0.17489 |
| EWT, 287 dokumen | 0.10068 | 0.11187 | 0.12611 | 0.12129 | 0.10575 | 0.11652 |

Berbanding full-softmax awal, perbezaan WikiText ialah +0.02938, bootstrap artikel 95% [0.02463, 0.03444]. Tiga seed +0.03177, +0.02971, +0.02666. EWT +0.02543 [0.01822, 0.03337], seed +0.02864, +0.02894, +0.01871. Kriteria peningkatan yang ditetapkan sebelum run lulus pada panel utama dan sekunder.

Berbanding margin, beza WikiText +0.00352 [0.00018, 0.00651] dan EWT +0.00481 [0.00170, 0.00793]. Bootstrap meresampel dokumen selepas purata tiga seed, bersyarat pada data dan seed yang dipilih. Berbanding checkpoint full-softmax awal yang dibetulkan semasa penilaian, beza kedua-dua panel kira-kira +0.01424. Ini memberi bukti bahawa melatih dengan offset tersebut membantu dalam persediaan ini, melebihi perubahan ranking pasca latihan semata-mata.

Vector10 ialah model yang telah dilatih dengan objektif margin, jadi jadual tidak membuktikan kelebihan seni bina Ising. Perbandingan seni bina memerlukan vector yang dilatih dengan objektif dan budget sepadan. Contoh yang lebih kerap boleh memberi MRR lebih tinggi tanpa membuktikan polysemy. EWT dan WikiText development telah digunakan berkali-kali, jadi keputusan masih eksploratori. Tiada ujian hardware atau sampler baharu.

## Semakan dan jejak

Logit disemak terhadap scorer asal, initial loss dan gradient finite, dan output latihan finite setiap batch. Evaluator menyemak pariti skor pada tetingkap dan calon sampel. Data dan checkpoint rujukan telah diaudit dalam `FULL_SOFTMAX_PILOT_RESULTS.md`. Setiap run merekod checkpoint, ranking, data hash, seed dan protokol ke MLflow `word-energy-full-softmax-prior-ablation`.

- Seed 17 `34ad840584164e5dac2059ca6ef5267b`.
- Seed 29 `6fd4829031424eba917b99c5fd1ab9ca`.
- Seed 43 `2ee7dde861fa4e82b960c0767e4bef6b`.
- Summary `b93a831d6d8248c1a1bdca5cf389fef0`.

Semua latihan menggunakan CPU dan penjumlahan tepat. Pilot ini tidak mengubah status Gaussian atau gate precision/rare events THRML.
