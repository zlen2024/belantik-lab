# Audit sementara critic: 7043

17 November 2026. GPT-6.1 Sol Ultra menyemak 16 run canonical 7043 dan
provenance retry selepas gangguan. Ini analisis interim; keputusan seluruh
panel bergantung pada baki 7019 dan 7027 yang masih dijalankan.

| Arm | Lulus joint100 |
|---|---:|
| Baseline | 0/4 |
| Early-bank | 2/4 |
| Late-bank | 0/4 |
| Early-song | 1/4 |

Early-bank gagal ambang practical rescue 7043 yang dibekukan (>=3/4), walaupun
median perbezaan paired bank entropy pada lexical100 +0.43730 nat memenuhi
manipulation check >=0.10. Bagi seed 251/257, tambahan entropy masing-masing
+0.7609/+0.7464 nat dan kedua-duanya lulus. Seed 263/269 mendapat +0.1215/
+0.1282 nat tetapi gagal; bank akhirnya tertumpu semula. Kenaikan sementara
entropy tidak menjamin sense atau pemisahan dua family yang betul.

Early-song pula menyelamatkan seed 269 walaupun bank entropy lexical100 hanya
0.1213, lebih rendah daripada early-bank seed sama (0.2379); bank entropy
kemudian meningkat dan sense akhir 1.0. Maka direct bank targeting dan bank
entropy awal bukan penjelasan tunggal yang mencukupi. Late-bank meningkatkan
entropy lexical200 tetapi gagal pada keempat seed; ini tidak membuktikan
bahawa model mustahil dipulihkan selepas itu, kerana daya gradient sebenar
bergantung pada keadaan model.

Critic menyemak data/schedule hashes, source, fixed handoff dan linkage:
attempt `b9af12890f824290b86ec81b7ac09d56` kekal KILLED, dengan retry
`77290528213f4d6cb3a5da8cf7c37188` sepadan bitwise kepada historical baseline.
Data ini selected-corpus synthetic oracle-assisted; tiada generalisasi bahasa
atau bukti physical chip. Pengiraan semakan ini dilakukan read-only.

## Pemerhatian tambahan 7027 seed 251

Baseline lulus, tetapi early-bank gagal. Critic mengesahkan initial parameters,
data, schedule dan handoff sepadan. Bank entropy pada lexical100 ialah 0.8001
untuk baseline berbanding 0.1597 untuk early-bank; pada joint100, 0.7675
berbanding 0.0127. Ini menunjukkan galakan lembut tidak menjamin entropy naik
sepanjang latihan bersama. Late-bank dan early-song kedua-duanya lulus pada
seed ini. Ini pemerhatian terpilih, bukan bukti mekanisme penyebab dan bukan
penilaian keseluruhan empat seed 7027.
