# Diagnosis validation Ising: tiga checkpoint

15 November 2026. Diagnosis eksploratori pada checkpoint terpilih sedia ada;
tiada retraining atau pemilihan checkpoint baharu. Region sense dibina daripada
training anchors; enam window F dan enam window R validation dinilai. Validation
yang sama digunakan dalam pemilihan model, jadi ini bukan ujian bebas.

| Seed | Run MLflow | Perbezaan anchor F/R (total variation) | F betul | R betul | Purata gain F / R | Joint ranking loss |
|---:|---|---:|---:|---:|---|---:|
| 17 | `4ab120cbd8af48b787d482a2d129e80f` | 0.99766 | 6/6 | 6/6 | +0.45286 / +0.54032 | 0.08154 |
| 29 | `aa7fdddb3d2146ec84c667f2b5e685f3` | 0.000807 | 6/6 | 0/6 | +0.002586 / −0.002586 | 0.39408 |
| 43 | `b1683c935a044a8e97576908599e79ad` | 0.99710 | 6/6 | 6/6 | +0.45229 / +0.54128 | 0.08001 |

Total variation 0 bermaksud distribusi anchor sama; 1 bermaksud terpisah
sepenuhnya. Ia mengukur perbezaan dua distribusi state, bukan accuracy bahasa.
Gain membandingkan probability region yang betul pada coupling 1 dengan
coupling 0, menggunakan checkpoint dan window yang sama.

Seed 29: q_F = `[0.990191, 0.004366, 0.004330, 0.001113]`, q_R =
`[0.989435, 0.004463, 0.005040, 0.001062]`, dalam urutan `--, +-, -+, ++`.
Kedua-dua anchor tertumpu pada `--`; prior bank juga 0.996825 pada `--`.
Pada validation, purata probability region betul ialah 0.999985 untuk F tetapi
hanya 0.0000145 untuk R. Context sedikit menaikkan F dan sedikit menurunkan R,
berbanding prior; ia tidak berjaya menukar keputusan pada contoh river.

Kesimpulan: bukti konsisten dengan anchor dan prior bank yang collapsed serta
respons context yang tidak memisahkan dua keluarga. Diagnosis ini tidak dapat
menentukan sama ada initialization, objective, atau optimization menyebabkan
collapse. Tiada kesimpulan tentang keberkesanan pada natural text atau hardware.
MLflow config artifacts menyimpan semua 12 baris, posterior setiap state,
checkpoint SHA-256 dan hash protocol. Ulang dengan
`python scripts/diagnose_ising_validation.py SEED`; log melalui
`scripts/log_ising_validation_diagnosis.py SEED`.
