# Keputusan scaling Word Energy, 20 November 2026

Kekalkan empat spin. Retrieval seluruh kamus dengan pembetulan kekerapan menghasilkan manfaat pada EWT. Hipotesis bahawa menggantikan satu pass latihan dengan tambahan 200k window daripada artikel sama akan menaikkan MRR Ising pada semua seed gagal pada WikiText.

## Urutan bukti

1. Readout pada parameter joint yang sama, WikiText 10,000 window, 60 artikel, 20 calon noise. Perubahan macro AUC +0.000785, interval 95% [-0.000479,+0.001987]. Run 3127680692bb423d8bdd780515b7fd65. Pada parameter latihan additive kesannya +0.000078 [-0.001076,+0.001225], run de8d4a33898c4cbd92c5880d09568df2. Critic Sol 6.2 ultra menyemak kedua-dua run dan theorem Lean. Tanda covariance menjelaskan perubahan satu skor, manakala ranking bergantung pada perubahan relatif antara dua calon.

2. WikiText, semua 2,000 calon. Skor asal memberi beza MRR purata artikel joint berbanding frequency prior -0.07848. Run 411d36760c164244900f1e901a72f85b. Susulan eksploratori dengan satu pekali tetap +log(noise_q) menaikkan beza joint berbanding prior kepada +0.01185 [0.00567,0.01836], dan joint berbanding additive pada parameter sama +0.00417 [0.00217,0.00588]. Run 782220fbb92b4f18ac410c53e0e44ff4. Kaedah dipilih selepas melihat validation, dan objektif margin tidak menjamin kalibrasi.

3. Replikasi dibekukan pada UD English EWT development r2.15, run 6219acde7e58415ba42ba9b067f7061e. Selepas penapisan pertindihan, 317/318 dokumen tersedia dan 287 menyumbang 10,000 window. Vocabulary meliputi 15,821/22,000 token (71.9%). Tiga seed menunjukkan keuntungan. Beza MRR purata dokumen joint vs prior +0.01555 [0.00902,0.02200], vs additive +0.00224 [0.00049,0.00397], vs vector10 +0.00477 [-0.00334,+0.01214]. Gate prior dan additive lulus. Ini retrieval pusat sebenar dalam window layak, dengan calon terhad kepada kamus latihan. Untuk reproduksi, simpan fail sumber di data/ewt/en_ewt-ud-dev.conllu, SHA-256 14a82ce2c4c4648c3e7c4efc76872c85c5f4c39643fcb7f9c91a45a58157a270. [Sumber EWT](https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.15/en_ewt-ud-dev.conllu)

## Pilot kepelbagaian data latihan

Semua cabang menerima 600k presentasi pada 600 artikel WikiText, kamus 2,000 dan tiga pass. A/A/A mengulang pool lama, A/B/A menggantikan pass kedua dengan 200k sampel tambahan. Kedudukan unik 141,101 dan 274,299. Setiap perbandingan berkongsi parameter dan Adam state selepas pass pertama. Rank cabang kawalan sepadan dengan ranking lama pada setiap window untuk dua panel. Selang percentile meresampel dokumen selepas mempuratakan tiga seed. Ia bersyarat pada latihan dan dokumen yang dikaji.

| Model dan panel | Beza MRR dokumen | Selang 95% | Seed 17 / 29 / 43 |
|---|---:|---|---|
| joint4_ewt | +0.001358 | [-0.000523, +0.003233] | +0.001582 / +0.000341 / +0.002151 |
| joint4_wiki | -0.001144 | [-0.003308, +0.000747] | +0.000954 / -0.002763 / -0.001624 |
| vector10_ewt | +0.004133 | [+0.001549, +0.007291] | +0.004838 / +0.001339 / +0.006221 |
| vector10_wiki | +0.000900 | [-0.001095, +0.002727] | +0.001245 / +0.000258 / +0.001198 |

## Jejak run

Summary 649129b36f944ded9cbad0092381c7a7. Setiap run mengandungi kedua-dua checkpoint akhir, hash keadaan bersama, rank setiap window, beza setiap dokumen dan protokol.
- joint4_17 1eb4520b22c24c5d95f73b627e3e77fb
- joint4_29 fa79f95f99ce4ca1b2cc4ae5cbc76e96
- joint4_43 ecc90b2a0dd74d6fac7d9b7bd11494b0
- vector10_17 e39982a310d344a7af4d0a6251799fe9
- vector10_29 3cf20c4471ea4e869999bd89c6fb54e3
- vector10_43 828009df8f054d94b397f82c0844128b

Percubaan perbandingan dengan checkpoint historical menggunakan syarat hash bit demi bit. Lima run lulus semakan permulaan dan dua percubaan joint seed 29 gagal. Beza maksimum pada percubaan diagnostik 6.73e-14. Keputusan utama menggunakan eksperimen shared-fork berasingan yang lengkap. Berikut semua percubaan awal.
- FINISHED vector10 seed 43 bc21b3e29d7a47698b8585aed235544e
- FINISHED joint4 seed 43 410ad86cbdb94e0290e28a823c65b796
- FINISHED vector10 seed 29 488646a85adc42d6b28027a2dde1f584
- FAILED joint4 seed 29 567b39bf8c6348088b38b0f4660bd2c0
- FAILED joint4 seed 29 a18b4ccdd44a4f1f88377e0f4d4bb357
- FINISHED vector10 seed 17 d227516aab4c49419e92ede83b84daa0
- FINISHED joint4 seed 17 73ab0e49429343179d7e34f1b8ef7a11

## Implikasi dua pelan

Ising: kekalkan 4 spin dan prior yang dibekukan sebagai baseline retrieval. Bukti setakat ini membezakan manfaat readout daripada manfaat tambahan sampel. Eksperimen susulan yang lebih bermaklumat ialah objektif ramalan pusat seluruh kamus pada data tetap, dibandingkan dengan latihan margin semasa, sebelum menambah kapasiti. Untuk sasaran chip, gate THRML empat-spin terdahulu masih gagal pada precision/rare events.

Gaussian: kekalkan D=1/M=2 sebagai rujukan. Audit sampler classical lulus meliputi semua 18 window validation dan tiga seed. Toy sense accuracy mencapai siling juga untuk lexical dan vector. Langkah seterusnya ialah set contoh konteks yang memerlukan kedua-dua petunjuk, dengan kawalan satu petunjuk dan additive. Tiada latihan Gaussian baharu dalam pilot ini.

Ujian rasmi reserved kekal tertutup. Semua kerja eksperimen di CPU tempatan. Gate yang berlainan menggunakan metric dan set calon berbeza.
