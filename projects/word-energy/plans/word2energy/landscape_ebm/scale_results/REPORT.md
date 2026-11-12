# Scaling Word2Vec-style dan energy neural pada text8

10 November 2026 — laporan awal, eksperimen CPU. Latar sumber primer dalam [WORD2VEC_SCALING.md](../WORD2VEC_SCALING.md); protokol awal [SCALE_WORD2VEC_PLAN.md](../SCALE_WORD2VEC_PLAN.md). Pemetaan p-bit dan ujian cip belum dibuat.

## Apa yang ditingkatkan

Eksperimen lepas: 10 juta aksara sumber train, 2,000 perkataan, 64 dimensi, 200,000 pasangan. Run ini: 50 juta aksara sumber train (8,498,923 token), sehingga 10,000 perkataan kerap minimum 5 kali, 100 dimensi, 1 juta pasangan tersampel dengan penggantian, diulang 3 epoch; kira-kira 3 juta pasangan dipersembahkan kepada setiap model. 10 blok train menyediakan 100,000 pasangan setiap satu. Sekitar 1,630,398 daripada 8,498,923 token dikekalkan selepas membuang kata kerap menggunakan kebarangkalian subsampling daripada paper (t=1e-5). Window maksimum 5, satu context disampel daripada window, dan 5 negatif mengikut q∝unigram^0.75. Ini corak yang diinspirasikan oleh Word2Vec, bukan replika implementasi asal: kita guna CPU PyTorch, SparseAdam/Adam, subset teks, pasangan statik disampel sebelum epoch dan hanya satu context setiap window sampel; hasil tidak setanding secara langsung dengan paper.

Kedua-dua model menggunakan sampel dan q sama; SGNS 2,000,000 parameter, neural 2,038,657. Model berbeza dalam bentuk energy, termasuk terma kuadratik neural, dan nilai awal rawak berbeza walaupun seed bernombor sama. Tiga seed model berkongsi data: ia bukan tiga korpus bebas. Val di aksara 70–75M dan test 75–80M, tidak digunakan pada eksperimen terdahulu mengikut rekod; bukan split standard. Tiada sempadan dokumen sebenar di text8. Train disubsample sebelum membina pasangan; val/test asal tidak disubsample. Vocab train-only meliputi 89.0% token test, dan anggaran 79.8% pasangan calon test kekal setelah syarat kedua-dua token dalam vocab.

## Penilaian utama — 1 juta pasangan

AUC dan hit@5 menguji pemeringkatan satu pasangan diperhatikan melawan 20 context rawak daripada q, bukan kefahaman semantik atau kebarangkalian bahasa. AUC rawak 0.5; hit@5 rawak 5/21≈0.238. Sesetengah negatif mungkin sebenarnya relevan, dan sekitar 0.424% negatif sama tepat dengan context positif. Skor ialah purata 3 seed dengan 5,000 contoh test sama; tiada CI merentas dokumen.

| Model | AUC test | Hit@5 test | BCE seimbang tidak dikalibrasi |
|---|---:|---:|---:|
| SGNS | 0.4880 | 0.2699 | 1.1788 |
| Neural energy | 0.5018 | 0.2756 | 1.1891 |

Checkpoint setiap model dipilih menggunakan validation AUC sebelum test utama dinilai. Semua model memilih epoch 3, tepi bajet, maka penumpuan tidak dibuktikan. BCE diukur dengan berat evaluasi 1:1 walaupun latihan 1:5; ia tidak dikalibrasi dan tidak boleh ditafsir sebagai NLL bahasa.

## Analogi dan liputan

Google questions-words mempunyai 19,544 soalan (8,869 semantik, 10,675 sintaks); hanya 4,936 boleh dinilai apabila semua empat kata berada dalam 10k vocab (770 semantik, 4,166 sintaks; liputan keseluruhan 25.26%). Skor cosine atas **vektor input** sahaja, bukan energy bersyarat neural:

| Model | Bilangan betul daripada 4,936, tiga seed | Purata accuracy pada yang diliputi |
|---|---|---:|
| SGNS | 15, 17, 14 | 0.31% |
| Neural | 8, 9, 6 | 0.16% |

Nilai rendah ini tidak patut dipanggil setara paper; data lebih kecil, latihan singkat dan variasi implementasi serta liputan berbeza. Tidak boleh guna analogy untuk menyatakan model membezakan makna `bank` dalam ayat.

## Diagnostik perbezaan taburan

Selepas melihat validation tanpa subsampling yang rendah, kita jalankan diagnostik **post-hoc pada validation sahaja**, memilih token dengan kadar subsampling train dan membina window semula. AUC val asal vs subsampled, purata 3 seed:

| Model | Val asal | Val subsampled semula |
|---|---:|---:|
| SGNS | 0.4512 | 0.6552 |
| Neural | 0.4676 | 0.6347 |

Ini menyokong bahawa definisi pasangan dan taburan mempengaruhi ukuran; ia mengubah token dan tingkap serentak, jadi bukan bukti sebab tunggal atau skor test. Model/epoch tidak dipilih semula menggunakan diagnostik ini. `val_subsampled_diagnostic.json` menyimpan hasil setiap seed. Jangan banding secara langsung nilai run ini dengan ujian terdahulu: data, split, vocab, dimensi, optimizer, subsampling dan sampel berbeza.

## Pengesahan dan fail

Set ujian analogi diperoleh daripada [repositori Word2Vec asal](https://github.com/tmikolov/word2vec/blob/master/questions-words.txt). Ujian `tests/test_scale_word2vec.py` menyemak sampler tanpa merapatkan OOV, backward sparse, dan pengiraan analogi serta liputan; 3 lulus. `results.json` mengandungi per seed, val/test, analogi dan masa; `noise_q.npy`, vocab dan checkpoint disimpan. Run utama mengambil kira-kira 495 saat pada CPU; sumber text8 MD5 disahkan. Laporan ini tidak menyimpulkan bahawa scaling telah mencapai prestasi Word2Vec penyelidikan atau menghasilkan minima bermakna. Susulan 5 juta pasangan telah selesai: lihat [laporan susulan](../scale_extended_results/REPORT.md).
