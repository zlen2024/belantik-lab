# Semakan bebas: factorial text8

17 November 2026. Critic GPT-6.1 Sol Ultra menyemak keputusan secara bebas selepas 12 run lengkap.

- Semua 12 run FINISHED. Semua 48 checkpoint hashes, 11 hashes array dataset, 11 source snapshots dan paired initial clones sepadan.
- NumPy scorer berasingan menggunakan pengiraan 16-state log partition secara langsung, bukan implementasi `expm1` yang digunakan trainer. Pengiraan semula epoch3 memberikan beza maksimum AUC 1.11e−16, MRR 0 dan validation objective 4.44e−16.
- Kawalan interaksi sifar tanpa offset: AUC 0.5, MRR 0.17358850975. Dengan offset: AUC 0.660195, MRR 0.26578727301.
- Mean AUC/MRR: tied 0.639893/0.274456; tied-frequency 0.698033/0.331214; role 0.678693/0.314663; role-frequency 0.700383/0.342324. Paired effects dan tanda setiap seed sepadan laporan.
- Interaksi negatif konsisten dengan manfaat kedua-dua perubahan yang bertindih dalam panel ini; ia tidak membuktikan mekanisme yang berdiri sendiri. Offset digunakan dalam training dan readout, jadi jangan panggil kesannya perubahan readout sahaja.
- Tiada frequency-offset shuffled control. Skor mengatasi frequency-only menunjukkan tambahan discrimination terhadap noise dalam ujian ini, bukan semantik yang betul. Perbandingan model vector/SGNS lama bukan perbandingan terkawal untuk explicit prior tersebut. Tiada tuntutan sense, symmetry-only, THRML parity atau cip.

Keputusan semakan: tiada kesilapan pengiraan atau blocker dikenal pasti; had di atas perlu dikekalkan dalam laporan dan kajian susulan.
