# Audit akhir critic: galakan entropy sementara

17 November 2026. GPT-6.1 Sol Ultra mengaudit 48 run canonical dan satu
attempt KILLED yang dipautkan kepada retry. Semua kombinasi selesai; 12
baseline sepadan bitwise dengan checkpoint dan metrik historical. Hash data,
schedule, initial parameters, fixed handoff, konfigurasi, sumber trainer,
dependency dan execution tier sepadan. Dua versi arm runner berbeza hanya
pada metadata retry. [Nota sambungan](ISING_ENTROPY_RESUMPTION_20261117.md)
merekod gangguan pengguna dan penyimpangan had masa satu batch asal.

Critic mengira semula 96 endpoint lexical400/joint100 menggunakan NumPy:
sense dan AUC tepat, ralat maksimum gain/TV/joint loss <=7.77e-16, dan tiada
percanggahan keputusan gate. Semua snapshot entropy bank/song diperiksa
semula. Bonus hanya digunakan dalam lexical-update wrapper; auxiliary loss
joint kekal pure ranking. Semakan ini dilakukan read-only; skrip NumPy bebas
untuk audit 96 endpoint tidak disimpan, tetapi data dan snapshots asal kekal
dalam MLflow.

| Permutasi | Baseline | Early-bank | Late-bank | Early-song |
|---:|---:|---:|---:|---:|
| 7043 | 0/4 | 2/4 | 0/4 | 1/4 |
| 7019 | 4/4 | 4/4 | 4/4 | 4/4 |
| 7027 | 2/4 | 2/4 | 2/4 | 2/4 |

Practical rescue gagal kerana 7043 memerlukan >=3/4. Manipulation check 7043
lulus: median paired bank entropy lexical100 meningkat 0.43730 nat. Timing
check pada endpoint 7043 lulus (early-bank 2/4 berbanding late-bank 0/4),
tetapi equal coefficient area tidak menjamin equal gradient force; kegagalan
late tidak membuktikan keadaan lewat mustahil dipulihkan. Early-song 7043
1/4 berbanding early-bank 2/4 tidak membuktikan target bank satu-satunya
mekanisme. Lihat [audit interim](ISING_ENTROPY_7043_INTERIM_CRITIC.md).

Perbandingan paired awal-bank berbanding baseline menunjukkan **3 pasangan
bertambah baik, 1 merosot dan 8 tidak berubah** daripada 12 pasangan seed.
Khusus pada 7027, jumlah kedua-dua arm sama 2/4 tetapi seed 251 merosot dan
seed 263 pulih. Early-song membantu 7043 seed 269; late-bank tiada perubahan
gate berpasangan. Jangan menggantikan analisis paired dengan jumlah kasar.

Cadangan critic: jadual dan kekuatan entropy yang diuji **tidak layak menjadi
default yang boleh dipercayai**. Jangan automatik menala λ atau memilih hasil
yang baik selepas melihat validation. Kajian ini ialah pembangunan posthoc
pada corpus synthetic terpilih dengan oracle, tiada reserved test, natural
text atau execution THRML/physical chip; tiada bukti kausal umum.
