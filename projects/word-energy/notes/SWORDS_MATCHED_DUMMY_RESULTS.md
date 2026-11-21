# Kawalan candidate dummy: SWORDS development

21 November 2026. Frozen protocol/code commit `312a0d8`, MLflow run `ed72133747a04946a7374e621838c1f7`. Ini data development yang sudah berulang kali diperiksa; official test tidak dibuka. Checkpoint tidak dilatih semula. Setiap candidate dinilai dalam sistem berasingan, dengan candidate lain diganti dummy uniform dan jumlah leaf serta normalisasi dikekalkan.

| Bacaan | AUC purata 3 seed |
|---|---:|
| Dummy dipadankan | 0.56737 |
| Semua candidate aktif bersama | 0.57568 |
| Candidate satu demi satu, skala lama | 0.58132 |
| Vector contextual | 0.62272 |

Dummy minus shared ialah -0.00831, source-proxy bootstrap interval [-0.01681, 0.01660]; dummy minus candidate-specific lama -0.01395 [-0.03838, 0.00657]. Kedua-duanya merangkumi sifar dan arah berbeza antara seed. Selepas buang sumber dominan, tinggal 55 target: dummy 0.53053, shared 0.52179, candidate-specific 0.54197, vector 0.58660. Keputusan tidak mengenal pasti punca utama secara meyakinkan; interval deskriptif bukan bukti equivalence. Model masih memerlukan latihan yang sejajar dengan konteks dan evaluation baharu. Tiada Gaussian, THRML atau chip diuji.

Perhatian: dalam `summary.json` MLflow, nama `shared` bermaksud dummy dipadankan dan `uniform` bermaksud shared asal, seperti field `mapping`; nama ini diwarisi daripada fungsi ringkasan dan bukan nama kondisi sebenar. Rujuk mapping sebelum bandingkan metrik.

Critic GPT-6.2 Sol medium menyemak protokol, code dan ringkasan: dummy uniform memberi mesej neutral kepada center, jadi kawalan mengekalkan divisor `m+K` sambil membuang identiti calon lain. Ia mengenal pasti skrip tidak mengesahkan sendiri digest manifest, penjajaran order candidate dan semua nilai finite; hash checkpoint diperiksa oleh loader tetapi hash dependencies tidak disimpan secara lengkap dalam run ini. Oleh itu keputusan ini preliminary dan perlu disemak semula sebelum dijadikan asas claim yang lebih kuat. Perbezaan arah mengikut seed dan sumber tidak membuktikan tiada kesan. Tiada hasil Lean baharu didakwa: semakan critic masih berjalan ketika catatan ini dibuat; bukti neutraliti sebelum ini ada dalam `SharedStarCriticAudit.lean`.

Cadangan critic: seterusnya bina protokol ranking konteks dengan graf sparse berdegree tetap, data dokumen yang diaudit dan split bebas, latihan Ising sejajar tugasan, serta perbandingan Gaussian klasik dan vector yang sepadan. Kawalan dummy ini bukan seni bina untuk deployment dan bukan eksperimen Gaussian.

Semakan bebas critic selepas itu membina semula marginal 16×16 tanpa scorer production: semua 9,114 skor dipadankan dalam `5.55e-16`, AUC per target tepat, digest manifest `210de7550edae7629e7360ad2b3e60b063146dbd430b7b0a3045ec2093818cd8` sepadan dan identiti target/source disahkan. Ini mengurangkan kebimbangan kesilapan pengiraan bagi run ini; kelemahan guard dalam skrip kekal.

Critic kemudian mengesahkan satu lemma baharu secara sementara dalam Lean: faktor bukan sifar yang sama pada semua 16 weight terbatal selepas normalisasi. `lake env lean --stdin` exit 0 pada Lean 4.36.0 dan Mathlib `fc4f59e577feb1a4956671bc04ac18aaaf8cd842`; axioms `propext`, `Classical.choice`, `Quot.sound`, tanpa `sorryAx`. Lemma ini memeriksa identiti matematik, bukan semantik atau hardware.
