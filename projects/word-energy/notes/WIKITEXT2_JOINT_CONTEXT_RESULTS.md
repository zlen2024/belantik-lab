# WikiText-2: latihan tenaga bersama tiga perkataan

19 November 2026. Eksperimen development [protokol](WIKITEXT2_JOINT_CONTEXT_PROTOCOL.md) commit `65397d5`: satu center dan dua context occurrences, 200,000 latihan dan 10,000 validation windows. Model Ising joint mengira six-spin star secara exact; `pair_add` menggunakan dua graph empat-spin independent dengan coupling per spoke sama; vector ialah shared 3D CBOW. Candidate center/noise, frequency prior, loss dan budget sama. Satu arm Ising joint lain dilatih dengan context bundles diacak. Semua tiga seed berkongsi data dan validation.

## Center prediction: epoch3

| Model | Parameter | AUC micro | AUC macro artikel | MRR@21 |
|---|---:|---:|---:|---:|
| joint | 6002 | 0.69270 | 0.69348 | 0.31898 |
| pair_add | 6002 | 0.69260 | 0.69258 | 0.31732 |
| vector | 6000 | 0.70025 | 0.70116 | 0.32636 |
| joint_shuffled | 6002 | 0.64823 | 0.64770 | 0.28621 |
| frequency-only | 0 | 0.65359 | 0.65133 | 0.28574 |

AUC membandingkan center sebenar dengan 20 calon noise; macro merata-ratakan 60 artikel, bukan penilaian sense atau likelihood.

| Perbandingan | Perbezaan micro seed 17 / 29 / 43 | Perbezaan macro | Bootstrap artikel bersyarat 95% |
|---|---|---:|---|
| joint_minus_joint_shuffled | +0.04623 / +0.04570 / +0.04150 | +0.04578 | [+0.03674, +0.05371] |
| joint_minus_pair_add | +0.00014 / +0.00043 / -0.00027 | +0.00091 | [-0.00088, +0.00264] |
| joint_minus_vector | -0.00768 / -0.00699 / -0.00796 | -0.00767 | [-0.01141, -0.00415] |

Joint mengatasi shuffle, jadi model memanfaatkan association center/konteks dalam ujian ini. Joint hampir sama dengan `pair_add` pada AUC: perbezaan micro berubah tanda dan interval macro merangkumi sifar. Pada MRR, joint meningkat sedikit dalam semua seed (+0.00193 / +0.00047 / +0.00258; purata +0.00166). Vector sepadan bilangan parameter lebih tinggi dalam tiga seed. Ini tidak menunjukkan manfaat yang jelas daripada perkongsian state center berbanding additive pada tugas/budget ini; parameter sama bukan jaminan kapasiti atau optimizer sama. Juga tidak menolak semua bentuk model joint lain.

## Diagnostik parameter

Bagi tiga run joint terlatih, purata mutlak perbezaan skor joint vs additive apabila dinilai pada parameter yang sama ialah sekitar 0.139–0.149; TV posterior center vs unary sekitar 0.200–0.209. Jadi graph tidak semestinya algebraically additive atau ignor konteks, walaupun AUC kedua-dua model terlatih hampir sama. Kedua-dua coupling menghampiri had 1.5 (sekitar 1.492–1.496) bagi semua seed; kemungkinan saturation ialah petunjuk diagnostik, bukan punca yang dibuktikan. Perubahan score apabila konteks kedua diganti dengan yang pertama juga direkod dalam setiap run; itu intervention deskriptif, bukan gold-sense counterfactual.

## Probe WiC berulang

[Protokol WiC berulang](WIKITEXT2_JOINT_WIC_PROBE_PROTOCOL.md) menggunakan subset development 167/638 yang **sudah dilihat sebelumnya**; tiada WiC label digunakan untuk melatih model. Ia bukan ujian benchmark bebas. Bacaan F3 menggunakan 1–4 context di WiC, walaupun train menggunakan tepat dua; additive-trained parameters menggunakan joint readout sebagai diagnostik selepas latihan.

| Readout | AUC purata | Seed 17 / 29 / 43 | Bootstrap target 95% |
|---|---:|---|---|
| joint trained | 0.54362 | 0.56399 / 0.54218 / 0.52468 | [0.4687, 0.6247] |
| pair-add trained, joint readout | 0.57470 | 0.59857 / 0.54978 / 0.57575 | [0.5025, 0.6539] |
| vector trained | 0.55519 | 0.54347 / 0.56657 / 0.55552 | [0.4803, 0.6332] |
| joint initial | 0.54892 | 0.54505 / 0.55323 / 0.54849 | [0.4885, 0.6142] |
| joint shuffled | 0.52076 | 0.51851 / 0.51621 / 0.52755 | [0.4464, 0.6067] |
| Jaccard konteks | 0.56377 | 0.56377 | [0.4845, 0.6442] |

Perbezaan joint−pair_add WiC purata -0.03109, bootstrap target bersyarat [-0.05463, -0.01019]; kedua-duanya memakai joint readout tetapi parameters dilatih dengan objective berbeza. Interval ini pada validation yang sudah digunakan dan merupakan salah satu daripada beberapa perbandingan eksploratori, bukan confirmatory atau sebab untuk memilih model menggunakan label. Semua batas vocabulary, topik, source overlap dan context-length shift masih terpakai.

## Setiap run latihan

| Arm | Seed | Run ID | AUC micro | AUC macro | MRR | Masa (s) |
|---|---:|---|---:|---:|---:|---:|
| joint | 17 | `d4ac69067db94b9ba6e9bee33e6d35da` | 0.69365 | 0.69337 | 0.31835 | 15.9 |
| joint | 29 | `5459599bea1042d7ba4cc6cc9dd5e8e7` | 0.69411 | 0.69545 | 0.32127 | 19.9 |
| joint | 43 | `1ca78c0254ef45688c6ab4dc850c99d9` | 0.69036 | 0.69164 | 0.31731 | 15.7 |
| pair_add | 17 | `cb6a51e8a6c045fe8d8c9d7600a800f0` | 0.69351 | 0.69197 | 0.31642 | 16.2 |
| pair_add | 29 | `df570c9a41524b67865f761176181693` | 0.69368 | 0.69385 | 0.32080 | 17.8 |
| pair_add | 43 | `92eba2d279114bb7b9da63a4c543a052` | 0.69063 | 0.69191 | 0.31473 | 17.3 |
| vector | 17 | `238d4bdaa10b4d949c4cf7c5b162bc12` | 0.70133 | 0.70141 | 0.32668 | 11.1 |
| vector | 29 | `d4b280542ad84888983e65f6d96c2d46` | 0.70110 | 0.70333 | 0.32570 | 11.7 |
| vector | 43 | `073d6ac07df149a8ade0be8ec257e527` | 0.69832 | 0.69873 | 0.32669 | 11.7 |
| joint_shuffled | 17 | `dccc22087adb4af19e1c3d5a84a4bbaa` | 0.64742 | 0.64728 | 0.28595 | 18.9 |
| joint_shuffled | 29 | `c9288b36529a48b2bb39729fe6a25fc2` | 0.64840 | 0.64768 | 0.28644 | 15.9 |
| joint_shuffled | 43 | `06ec285b31e44e3a8aae818fd6378788` | 0.64886 | 0.64815 | 0.28624 | 15.7 |

## Data dan had

- Asal sampel daripada dataset `b37da8076fd54f4da389082278e9733a`. Eligible sampled-pair rows: 177,060 train dan 8,766 validation. Windows unik occurrence: 143,233 train/7,162 validation, melibatkan 600/60 artikel. Resampling daripada pasangan sedia ada bermakna bukan sampel uniform baharu daripada semua center corpus.
- Context-bundle shuffle mengekalkan center dan pasangan context; collisions noise dikekalkan. Prior center, 3 epoch, lima train noise/20 validation noise dan canonical unique triple score sama. Keadaan satu perkataan muncul dua kali menggunakan spin occurrence berasingan.
- Model joint ialah Ising star yang boleh diperiksa exact secara klasik; tiada THRML parity untuk graph terlatih enam-spin ini, compiler/hardware, kos tenaga fizikal atau sense labels untuk train diuji. Tidak membuktikan wujud dua lembah makna.
- WiC validation telah diperiksa dan digunakan sebagai development; test WiC, WikiText dan reserved text8 tidak digunakan untuk latihan atau penilaian. Wikipedia boleh bertindih antara corpus; artikel tidak semestinya independent sepenuhnya.

MLflow experiment `word-energy-wikitext2-joint-context`: data `f1b7db5231ed4a5788b566e388f8e731`, lexical summary `29469e5750ab490eb6e8f4d2650cbeb8`, repeated WiC `48704618227d4d98b98056908c3b5591`. Data arrays, source hashes, initial dan semua 36 epoch checkpoints, scores, diagnostics dan setiap prediction disimpan dalam MLflow tempatan.

Semakan bebas dan kritikan hipotesis: [laporan critic](WIKITEXT2_JOINT_CONTEXT_CRITIC.md).
