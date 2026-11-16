# Audit critic: panel permutasi corpus 227

16 November 2026. Critic GPT-6.1 Sol Ultra mengaudit 32 run MLflow:
identiti dan status, data dan schedule, susunan sebenar bias, pasangan
parameter awal, konfigurasi kawalan asal dan handoff. Tiada ketidakpadanan
ditemui. Ringkasan lulus joint100 ialah 3, 3, 4, 2, 3, 0, 1 dan 3
daripada empat seed; lexical400 memberi keputusan lulus yang sama untuk
setiap run. Kiraan NumPy bebas terhadap 64 endpoint lexical400 dan joint100
mempunyai ralat maksimum 8.23e-16 dan tiada percanggahan keputusan gate.
Peak working set yang direkod ialah 545.125 MiB.
Kiraan critic dilakukan secara read-only; skrip enumeration bebas itu tidak
disimpan. Snapshot asal dan metrik tetap tersedia dalam MLflow.

Analisis di bawah bersifat **posthoc**, pada satu corpus yang dipilih selepas
keputusan asal diketahui. Purata entropi bank dan anchor total variation
(TV) bagi empat seed pada joint100:

| Permutasi | Lulus | Bank entropy (nats) | Anchor TV |
|---:|---:|---:|---:|
| 7011 | 3/4 | 0.61510 | 0.75679 |
| 7013 | 3/4 | 0.58318 | 0.76316 |
| 7019 | 4/4 | 0.77606 | 0.99737 |
| 7027 | 2/4 | 0.39095 | 0.49965 |
| 7039 | 3/4 | 0.58956 | 0.85416 |
| 7043 | 0/4 | 0.01489 | 0.00550 |
| 7057 | 1/4 | 0.22738 | 0.45822 |
| 7069 | 3/4 | 0.58687 | 0.77496 |

Perjalanan purata empat seed; setiap sel ialah entropy bank dalam nats /
train-anchor TV. Purata boleh mencampurkan seed yang lulus dan gagal.

| Permutasi | Initial | Lexical20 | Lexical100 | Lexical400 | Joint100 |
|---:|---:|---:|---:|---:|---:|
| 7011 | 1.2580 / 0.1434 | 1.2078 / 0.4618 | 0.5349 / 0.7223 | 0.6282 / 0.7567 | 0.6151 / 0.7568 |
| 7013 | 1.2854 / 0.1018 | 1.0273 / 0.3503 | 0.4700 / 0.5890 | 0.5873 / 0.7702 | 0.5832 / 0.7632 |
| 7019 | 1.2832 / 0.0834 | 1.2465 / 0.4109 | 0.8110 / 0.9653 | 0.7788 / 0.9946 | 0.7761 / 0.9974 |
| 7027 | 1.2705 / 0.1065 | 0.8575 / 0.2758 | 0.3651 / 0.4778 | 0.3973 / 0.4984 | 0.3909 / 0.4996 |
| 7039 | 1.2301 / 0.1033 | 1.0243 / 0.3569 | 0.2871 / 0.6493 | 0.5945 / 0.8729 | 0.5896 / 0.8542 |
| 7043 | 1.2705 / 0.1347 | 0.8448 / 0.2565 | 0.0947 / 0.0457 | 0.0311 / 0.0082 | 0.0149 / 0.0055 |
| 7057 | 1.2301 / 0.0989 | 0.8560 / 0.3216 | 0.2143 / 0.4286 | 0.2169 / 0.4834 | 0.2274 / 0.4582 |
| 7069 | 1.3038 / 0.0738 | 1.1028 / 0.2541 | 0.4090 / 0.5435 | 0.5939 / 0.7772 | 0.5869 / 0.7750 |

Bagi 7043, bank dan kedua-dua keluarga F/R cenderung menumpu pada state yang
sama; 7019 mengekalkan dua keadaan berbeza. Untuk seed 251, kebarangkalian
bank pada state utamanya ialah 0.99823 pada 7043, dengan purata anchor F dan R
0.98444 dan 0.99025 pada state itu. Sebaliknya 7019 meletakkan sekitar
0.53408 dan 0.44714 pada dua state berbeza. Entropi bank 7043 sudah rendah
pada lexical100, sebelum joint training. Ini menerangkan corak *collapse*,
bukan membuktikan punca.

Jarak centroid F–R awal ialah 0.11609 untuk 7043 berbanding 0.04772 untuk
7019. Jadi jarak awal yang lebih besar sahaja tidak menjamin hasil lebih baik.
Kedua-dua kumpulan mendapat AUC 1.0: metrik tersebut tidak mengenal pasti
kegagalan pemisahan F/R dalam toy ini. Tiada bukti hard parameter-bound
saturation daripada titik semakan yang diperiksa. Corak anchor collapse 7043
bukan satu-satunya bentuk kegagalan: contohnya 7057 seed 269 gagal sense walau
anchor TV 0.8143. Semua 19 run lulus mempunyai entropi bank akhir 0.7675–0.9022
nats, berbanding 0.01265–0.11387 bagi 13 run gagal. Hubungan ini deskriptif dan
tidak diuji sebagai pemilih checkpoint atau peraturan keputusan baharu.
Lihat [hasil penuh](ISING_CORPUS227_PERMUTATION_PANEL_RESULTS.md).
