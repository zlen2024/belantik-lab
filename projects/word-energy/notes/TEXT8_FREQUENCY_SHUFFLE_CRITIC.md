# Semakan bebas Sol: text8 frequency shuffle

18 November 2026. Critic GPT-6.1 Sol Ultra menyemak reka bentuk sebelum run dan mengaudit semula data serta hasil selepas run secara read-only.

- Enam run shuffled FINISHED, dipadankan dengan enam checkpoint asal. Hash kesemua 48 checkpoint, 11 arrays dataset, initialization berpasangan dan source code teras sepadan.
- Scorer NumPy bebas menggunakan direct 16-state log partition. Perbezaan maksimum daripada metrik epoch3 yang direkod: AUC 1.11e−16, MRR 0, objective 4.44e−16.
- AUC asal − shuffled seed 17/29/43: tied-frequency +0.04665/+0.03608/+0.04579 (purata +0.04284); role-frequency +0.04803/+0.04620/+0.04883 (purata +0.04769). Kedua-duanya melepasi kriteria deskriptif dibekukan.
- Subgroup 18,440 comparisons dengan ratio count antara 0.5 dan 2: perbezaan purata +0.09548 tied dan +0.09226 role. Subgroup exact-count 217 comparisons: +0.11674 dan +0.06605. Semua perbezaan per seed positif, tetapi subgroup sekunder dan comparisons berkorelasi.
- Shuffled AUC purata sedikit di bawah baseline frequency-only 0.660195: sebahagian jurang original−shuffled mungkin menunjukkan kemerosotan kontrol, bukan hanya useful context signal. Semua original seed melebihi baseline tersebut. Frequency-only AUC subgroup luas 0.523617 dan exact-count 0.5.
- Hasil menyokong manfaat observed pair association di bawah frequency prior dan prosedur ini, bukan bukti kefahaman semantik. Satu permutasi mengekalkan empirical sampled context marginal dan masih boleh belajar marginal corrections/associations rawak; shuffled tidak semestinya chance atau frequency-only. Tiga seed tidak menguji variasi corpus atau shuffle.

Tiada blocker pengiraan dikenal pasti. Langkah seterusnya yang dibatasi: kekalkan panel ini dan rancang validation baharu dengan beberapa permutasi serta kawalan model vector yang menerima prior sama, tanpa membuka reserved test untuk tuning.
