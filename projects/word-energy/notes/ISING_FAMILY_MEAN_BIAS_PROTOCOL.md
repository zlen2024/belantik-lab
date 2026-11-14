# Cadangan: reciprocal family-mean bias transfer

15 November 2026. Cadangan selepas sintesis per-anchor dan critique 6.1 Sol
Ultra. **Belum dijalankan.** Ini hipotesis validation-only yang terbatas pada
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
