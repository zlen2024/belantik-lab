# Protokol pertukaran bias anchor sahaja

15 November 2026. Selepas semakan critic 6.1 Sol Ultra terhadap transplant
anchor penuh, uji sama ada 32 raw bias bagi 16 anchor F/R mencukupi untuk
memindahkan hasil antara background seed29/43. Raw internal coupling setiap
anchor dan seluruh parameter bukan anchor dikekalkan daripada background.

Dua reciprocal swaps; original dan full anchor swaps ialah controls. Fixed
lexical400, optimizer/schedule/negatives asal, validation exploratory, tiada
joint training atau test. 800 updates, empat CPU threads; memory peak tidak diukur.
Jika hasil berpindah, donor internal coupling tidak diperlukan untuk pemindahan
yang diperhatikan di sini; jika tidak, internal atau interaksi masih terbuka.
Hasil dua seed tidak membuktikan punca unik atau generalisasi.

## Keputusan fixed lexical400

| Background | Bias donor | Anchor TV | Context TV | F / R betul | AUC | Lexical loss | MLflow |
|---:|---:|---:|---:|---|---:|---:|---|
| 29 | 29 | 0.001861 | 0.000000837 | 6/6 / 0/6 | 1.0 | 0.26280 | `44af1bc171e340bf8b6d9d4c02c8c191` |
| 29 | 43 | 0.995145 | 0.999235 | 6/6 / 6/6 | 1.0 | 0.17888 | `0fa55ff18422464593bc755f0aba849a` |
| 43 | 43 | 0.993516 | 0.992376 | 6/6 / 6/6 | 1.0 | 0.13753 | `b51e07f7fbb4432492fe2a6556049750` |
| 43 | 29 | 0.021374 | 0.000007660 | 6/6 / 0/6 | 1.0 | 0.37070 | `ca6d88b60cd14aeb9b76d5d5eb285c9d` |

Kejayaan/kegagalan berpindah bersama bias donor dalam pasangan seed ini.
Ini menunjukkan pertukaran **initial** internal coupling donor tidak diperlukan
untuk pemindahan tersebut; internal coupling tetap dilatih dan boleh penting.
Ia tidak membuktikan internal coupling tiada kesan secara
umum. Masa latihan tambahan 8.58 dan 8.70 saat. Hasil pada validation kecil
yang pernah digunakan untuk selection; bukan bukti generalisasi, gate asal
berjaya, atau execution thermodynamic hardware.

Critic 6.1 Sol Ultra memeriksa hashes dan mengira outcome secara bebas; tepat
32 raw bias anchor sahaja ditukar, manakala internal dan parameter lain bermula
daripada background asal. Semua snapshot hingga lexical400 wujud.
