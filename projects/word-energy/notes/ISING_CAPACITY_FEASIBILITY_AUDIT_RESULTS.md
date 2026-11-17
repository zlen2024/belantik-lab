# Keputusan audit feasibility dan latihan

17 November 2026. Ini audit posthoc 10 checkpoint pada corpus synthetic 227;
tiada training baharu. [Protokol](ISING_CAPACITY_FEASIBILITY_AUDIT_PROTOCOL.md)
mengandungi pilihan checkpoint, definisi metrik dan kawalan. MLflow experiment
`word-energy-capacity-feasibility-audit`, run
`a2d4245bb9ce4dc7ad964b5f6bf520f4`, menyimpan sumber, hashes, metrik dan
diagnosis penuh setiap checkpoint.

Critic GPT-6.1 Sol Ultra mengira semula 10 checkpoint secara bebas menggunakan
NumPy tree factorization yang mengintegrasikan setiap leaf melalui empat state.
Semua metrik, kawalan dan bound sepadan dengan ralat maksimum 5.0e-16; tiada
perbezaan keputusan gate. Hash checkpoint dan source historical diperiksa.
Pengiraan bebas ini dibuat read-only dan skrip critic tidak disimpan; kod audit
utama dan source snapshots tersedia dalam MLflow.

## Bukti utama

| Checkpoint fixed joint100 | Sense | AUC | Context gain | Full train F2 | Cross F/R negative loss |
|---|---:|---:|---:|---:|---:|
| 7019 baseline, 4 seed | 1.0 semua | 1.0 semua | 0.4437–0.4451 | 0.1300–0.1308 | 0.2066–0.2111 |
| 7043 baseline, 4 seed | 0.5 semua | 1.0 semua | 0.00000007–0.00000351 | 0.2597–0.2599 | 1.3076–1.3103 |
| 7043 early-bank, seed 251/257 | 1.0 kedua-dua | 1.0 kedua-dua | 0.4442–0.4446 | 0.1288–0.1293 | 0.2081–0.2093 |

`7043` dan `7019` ialah permutasi bias awal; architecture, data, training
schedule, tensor shapes dan bounds sama. Initial internal/interaction bagi
paired seeds sepadan. Dua witness early-bank bermula dari keseluruhan raw
initial parameters yang sama dengan baseline 7043 berkenaan, tetapi
menggunakan regularizer sementara semasa latihan. Jadi **parameter yang
memenuhi gate memang wujud dalam model class yang sama untuk tugasan toy
ini**. Witness berjaya juga mempunyai objective training lebih rendah daripada
checkpoint gagal; training negatives masih menggunakan family/grammar oracle.

Ini menolak dakwaan bahawa struktur dua-spin itu secara mutlak tidak mampu
memenuhi gate ini. Ia tidak membuktikan sebab tunggal kegagalan: initialization,
training objective, optimizer dan schedule tidak diasingkan sepenuhnya.
Regularizer entropy terdahulu juga bukan penyelesaian reliable: hanya 2/4
seed 7043 lulus dan ia merosakkan satu seed 7027.

## Apa yang menghalang checkpoint gagal?

Dengan **unary parameters bank dan coupling dibekukan**, setiap checkpoint
7043 gagal mempunyai lower bound berikut bagi kebarangkalian posterior state
bank yang dominan, untuk sebarang dua context leaves dalam normalized F3:

| Seed | Lower bound untuk state dominan |
|---:|---:|
| 251 | 0.9233 |
| 257 | 0.9236 |
| 263 | 0.9268 |
| 269 | 0.9193 |

Ini bukan kebarangkalian diukur pada chip. Bound matematik daripada model
berkata state sama kekal sekurang-kurangnya sekitar 92% walaupun unary
parameters context berubah secara sewenang-wenangnya, selama bank dan coupling
tetap. Oleh itu pada checkpoint gagal ini, menukar perkataan context sahaja
tidak boleh membuat bank bertukar kepada state lain sebagai state paling
berkemungkinan. Ia **tidak** bermakna seni bina secara umum tidak mampu: ubah
parameter bank/coupling dan witness di atas membuktikan kemungkinan lain.

Pada checkpoint yang berjaya, nilai lower bound yang sama sekitar 0.055–0.059;
nilai kecil itu hanya lower bound yang longgar, bukan bukti state sebenar
berkebarangkalian 5%. Semua validation contexts yang diperiksa menghormati
bound tersebut.

## Kawalan dan batasan

Set coupling ke sifar memberi context gain sifar bagi kesemua checkpoint.
Menukar context F/R secara berpasangan sambil mengekalkan label membalikkan
accuracy/gain seperti dijangka. Kedua-duanya sanity check yang sebahagiannya
mengikut definisi; bukan bukti statistik bebas atau bukti makna sebenar.
Validation corpus ini sudah digunakan untuk development; F/R readout memakai
label anchor latihan. Tiada reserved-test scoring, natural text, THRML
sampling atau cip fizikal diuji.
