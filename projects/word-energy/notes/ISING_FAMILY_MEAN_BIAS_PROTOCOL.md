# Cadangan: reciprocal family-mean bias transfer

15 November 2026. Cadangan selepas sintesis per-anchor dan critique 6.1 Sol
Ultra. Dua reciprocal transfer telah dijalankan. Ini hipotesis validation-only yang terbatas pada
seed29/43 dan data synthetic yang family labelnya diketahui.

## Soalan

Adakah purata bias awal bagi setiap family F/R sahaja cukup untuk memindahkan
hasil yang diperhatikan pada full bias swap, tanpa menyalin corak setiap token?

## Intervensi dibekukan

Untuk token anchor `t` dalam family `f`, pada initialization sahaja:

`b_new[t] = b_background[t] + mean_trainweighted(b_donor, f) - mean_trainweighted(b_background, f)`.

Gunakan bilangan occurrence anchor dalam training windows sebagai weights seperti
sense map asal; dua komponen raw bias untuk setiap F/R family. Kaedah ini
memindahkan empat komponen purata tetapi mengekalkan residual setiap token
relatif kepada weighted mean asal. Kekalkan raw internal coupling semua token,
bank/song/M, interaction weights, dataset, negatives, schedule, Adam, lr,
clip dan 400 update lexical tepat seperti eksperimen asal. Jangan clip raw bias
selepas shift; lapor range serta physical transformed bias `2.5*tanh(raw)`.

| Run | Background | Donor family mean |
|---|---:|---:|
| A | 29 | 43 |
| B | 43 | 29 |

Gunakan original29/43 dan full anchor-bias reciprocal swaps sebagai controls.
800 optimizer updates tambahan, <=4 CPU threads; log elapsed dan ukur memory
peak jika tersedia. Fixed endpoint lexical400 serta snapshots 0,20,...,400;
no joint retraining, seed search atau test evaluation.

## Semakan sebelum mentafsir

- Verify donor weighted means sepadan dalam tolerance `1e-12` dan recipient
  residuals kekal dalam tolerance `1e-12` untuk setiap F/R token dan koordinat.
- Verify segala parameter lain sama tepat, dan replay asal telah melepasi gate.
- Report anchor TV weighted dan unweighted, within-/cross-family per-token TV,
  state probability setiap token, prior bank, context TV, per-family counts dan
  gain, lexical AUC/loss, full training lexical loss serta source/data hashes.
- Abort jika snapshot nonfinite, residual/mask salah atau provenance hilang.

Jika kedua-dua swaps memindahkan separability mengikut donor seperti full swap,
purata family itu mencukupi **untuk dua background dan budget ini**. Jika tidak,
residual individu atau interaksinya masih calon penjelasan; ia tidak membuktikan
setiap token penyebab. Intermediate/partial outcomes perlu dilapor berasingan.
Ini family-label-informed diagnostic, bukan cara deployment tanpa label, bukan
ujian pada seed bebas, bukan pembaikan tervalidasi atau bukti hardware.

## Hasil fixed lexical400

| Background / donor | Weighted anchor TV | Context TV | F / R betul | Validation AUC | Validation lexical loss | MLflow |
|---|---:|---:|---|---:|---:|---|
| 29 / 29 original | 0.001861 | 0.000000837 | 6/6 / 0/6 | 1.0 | 0.26280 | `44af1bc171e340bf8b6d9d4c02c8c191` |
| 29 / 43 family mean | 0.995296 | 0.999281 | 6/6 / 6/6 | 1.0 | 0.17883 | `b559a7bb1fce43848160af1e83f88e4d` |
| 43 / 43 original | 0.993516 | 0.992376 | 6/6 / 6/6 | 1.0 | 0.13753 | `b51e07f7fbb4432492fe2a6556049750` |
| 43 / 29 family mean | 0.043153 | 0.0000780 | 0/6 / 6/6 | 1.0 | 0.37178 | `96ac384afd1d477bb215d5545f4ec924` |

Kedua-dua hasil kategori berpindah bersama family mean donor pada dua seed
dan bajet ini. Ini konsisten dengan purata tersebut mencukupi untuk pemindahan
yang dilihat; residual individu tidak disalin. Intervensi tetap berasaskan label
family synthetic dan tidak menunjukkan penyelesaian yang general atau bahawa
pilot asal lulus.

Mean error maksimum `6.94e-18`, residual error maksimum `2.78e-17`; parameter
lain disahkan sama tepat. Raw bias range 29←43 ialah `[-0.15048,0.12891]`,
dan 43←29 `[-0.19284,0.19064]`; transformed ranges masing-masing
`[-0.37338,0.32051]` dan `[-0.47621,0.47090]`. Runtime latihan 8.97/8.58s;
OMP dan OpenBLAS ditetapkan empat thread; peak memory tidak diukur.
Bilangan thread CPU sebenar juga tidak diukur secara bebas.

Unweighted anchor TV 0.995162 / 0.055521; mean within-family pairwise TV
F/R 0.001227/0.001234 dan 0.155428/0.056240; cross-family mean TV
0.995465 / 0.100712. Artifact `config.json` menyimpan per-token probabilities,
family counts/gains, semua snapshots serta hashes sumber/data. Full training
lexical loss ialah 0.178651 / 0.375069; ia tidak sama dengan validation loss.

Critic 6.1 Sol Ultra mengesahkan secara bebas formula, mask, hashes dan metric
dengan enumeration NumPy. Hasil menyokong pemindahan kategori pemisahan
tinggi/rendah, bukan kesamaan numerik trajectory: tiada margin equivalence
numerik dipradaftarkan. Pada background43/donor29, family mean memberi F 0/6,
R 6/6, sedangkan full bias swap memberi F 6/6, R 0/6; kedua-duanya accuracy
0.5 tetapi readout terperinci berbeza. Keputusan tidak menentukan sama ada
kesan berpunca daripada common family offset atau kontras F–R. Critic mencadangkan
tutup protocol ini dengan penjelasan tersebut; tiada training tambahan diperlukan.
