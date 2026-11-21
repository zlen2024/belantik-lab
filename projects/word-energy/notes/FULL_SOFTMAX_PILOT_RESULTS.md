# Keputusan pilot full-vocabulary — 21 November 2026

## Soalan dan kaedah

Adakah latihan cross-entropy terhadap semua 2,000 perkataan meningkatkan retrieval berbanding latihan margin sedia ada? Tiga seed 17, 29 dan 43 menggunakan empat spin, data, inisialisasi dan 600,000 presentasi yang sama. Perbandingan utama menggunakan skor latihan cross-entropy dan skor margin dengan pembetulan noise yang dibekukan sebelum pilot. Oleh itu kesan utama meliputi kedua-dua objektif dan readout. WikiText dan EWT adalah development yang telah digunakan sebelum ini, bukan test baharu.

## Keputusan

MRR lebih tinggi menunjukkan perkataan sebenar cenderung berada pada kedudukan lebih atas. Angka ini ialah purata MRR per artikel, kemudian purata tiga seed.

| Panel | Full softmax, skor asal | Margin, skor dibetulkan | Beza | 95% bootstrap artikel |
|---|---:|---:|---:|---|
| WikiText, 60 artikel | 0.1686 | 0.1945 | -0.02586 | [-0.03224, -0.01953] |
| EWT, 287 dokumen | 0.1007 | 0.1213 | -0.02062 | [-0.02821, -0.01294] |

Perbezaan setiap seed pada WikiText ialah -0.02619, -0.02550 dan -0.02589. Pada EWT ia ialah -0.02188, -0.02295 dan -0.01702. Gate awal tidak lulus. Bootstrap adalah bersyarat pada artikel dan seed yang dipilih.

Pembetulan kekerapan sangat mempengaruhi keputusan. Untuk WikiText, purata full softmax naik daripada 0.1686 kepada 0.1838 apabila pembetulan margin yang sama digunakan selepas latihan. Margin dengan pembetulan mendapat 0.1945. Pada EWT, angka sepadan ialah 0.1007, 0.1119 dan 0.1213. Ini ialah analisis diagnostik menggunakan checkpoint yang sama, bukan ujian latihan semula dengan offset baru. Full softmax dengan pembetulan masih lebih rendah daripada margin bagi setiap seed pada kedua-dua panel.

## Semakan dan had

Critic GPT-6.2 Sol ultra menyemak objektif, confounder dan protokol. Logit calon disemak terhadap scorer asal, gradient diperiksa dengan finite difference, dan setiap ranking baseline dikira semula baris demi baris. Tiada perbezaan ranking pada enam gabungan seed-panel. Hash inisialisasi dan checkpoint asal, data dan panel juga disemak. Audit MLflow `c6d95c09ada74d02b3b6b4eb4d48c364`.

Setiap run baharu melibatkan 1.2 bilion skor calon latihan, berbanding kira-kira 3.6 juta positif dan negatif untuk margin. Bilangan skor tidak menentukan nisbah runtime kerana operasi berbeza. Latihan baharu mengambil 58.4, 62.0 dan 56.2 saat pada CPU dalam tiga run; masa margin sedia ada merangkumi aktiviti tambahan dan tidak setara. Tiada latihan atau pelaksanaan cip fizikal dilakukan. Keputusan retrieval tidak membuktikan polysemy.

Critic mengenal pasti prior 0.25 log-count yang digunakan semasa latihan cross-entropy meletakkan berat lebih rendah pada perkataan paling kerap berbanding kekerapannya dalam window latihan. Ini satu hipotesis untuk menerangkan sebahagian jurang, bukan sebab yang sudah dibuktikan. Susulan yang boleh menguji punca itu ialah melatih semula cross-entropy dengan offset log-count penuh pada data, model dan optimizer yang sama. Kawalan vector yang dilatih dengan objektif sama diperlukan sebelum membuat kesimpulan kelebihan seni bina.

Rekod MLflow `word-energy-full-softmax-pilot`: seed 17 `30c212437ae840bebf7e22fc8294c420`, seed 29 `1531b87bf41048ea89dc796970d007f4`, seed 43 `77636915e34a45768e7ea01da7d940a9`, summary `661da7b625644dccb5d49624baf02069`. Fail protokol telah ditambah butiran semakan critic semasa run sedang berjalan. Kandungan asal dan snapshot yang digunakan setiap run tersedia sebagai artifact.

Gaussian tidak dilatih dalam pilot ini. Cadangan critic untuk ujian berikutnya ialah contoh seimbang yang memerlukan interaksi dua petunjuk, dengan kawalan additive dan model vector yang juga mampu mempelajari interaksi. Semak dahulu bahawa Gaussian D=1/M=2 boleh mewakili tugasan tersebut pada fixture terkawal.
