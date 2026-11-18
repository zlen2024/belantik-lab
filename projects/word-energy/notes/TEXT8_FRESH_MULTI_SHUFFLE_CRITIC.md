# Semakan bebas Sol: fresh validation dan multi-shuffle

18 November 2026. GPT-6.1 Sol Ultra mengkritik protokol dan kod sebelum run, kemudian membuat audit read-only hasil menggunakan implementasi berasingan.

- Validation/noise dibina semula daripada archive, vocabulary/counts/q train dan seeds yang ditetapkan. Ketiga permutasi dibina semula. Semua 144 checkpoint (36 endpoint × 4 tahap), initialization, source dan identiti data berpasangan diperiksa.
- Scorer NumPy dengan direct 16-state Ising log partition serta vector dot mengira semula semua endpoint. Beza maksimum AUC 1.11e−16, MRR 5.55e−17 dan objective 4.44e−16 berbanding rekod.
- Frequency-only fresh: AUC 0.64652, MRR 0.25829699. Purata original AUC: tied 0.682005, role 0.68434167, vector 0.687685.
- Semua 9 perbezaan original−shuffle positif bagi setiap model; purata semua/baharu sahaja: tied +0.04147444/+0.042125, role +0.04664/+0.04709833, vector +0.04515111/+0.04588. Sel 3×3 berkongsi sample dan original checkpoint, bukan sembilan replikasi bebas.
- Vector mengatasi kedua-dua purata Ising sedikit; tied−vector −0.00568 dan role−vector −0.00334333. Vector lebih tinggi dalam dua daripada tiga seed bagi setiap perbandingan. Capacity dan parameter count berbeza; hasil belum membuktikan kelebihan umum mana-mana architecture.
- Hasil konsisten dengan manfaat observed pair association pada validation ini, tetapi kawalan acak yang jatuh di bawah frequency-only boleh membesarkan jurang. Ini belum mengasingkan semantik daripada statistik bahasa atau membuktikan physical chip.

Audit tidak menemui blocker pengiraan. Langkah seterusnya yang dibatasi: laporkan keputusan dan rancang ujian lebih luas dengan unit data bebas serta baseline sama prior, mengekalkan reserved test sehingga protokol betul-betul dibekukan.
