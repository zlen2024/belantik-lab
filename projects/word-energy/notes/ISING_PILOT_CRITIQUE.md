# Critique — Ising exact-ranking pilot

**Tarikh:** 15 November 2026  
**Jenis:** experiment critique + hypothesis critique  
**Status tindakan:** cadangan diagnosis read-only; belum retrain atau ubah protocol

## Fakta yang diukur

Pilot Ising exact-enumeration tidak lulus gate tiga-seed yang telah ditetapkan.
Semua checkpoint dipilih daripada validation sebelum test dibuka.

| Seed | Selected MLflow run | Validation/test sense | Context gain | Lexical AUC |
|---:|---|---:|---:|---:|
| 17 | `0706fb0f32504442a8e8de7b52214222` | 1.0 / 1.0 | 0.4966 | 1.0 |
| 29 | `4eb16727f30c48c5bd1b8a455534f6dc` | 0.5 / 0.5 | hampir 0 | 1.0 |
| 43 | `45b5fa9ee2a34c42972e16b484ed7298` | 1.0 / 1.0 | 0.4968 | 1.0 |

Pemeriksaan read-only checkpoint terpilih mendapati seed 29 mempunyai anchor
unary yang runtuh ke state yang sama:

```text
q_F = [0.99019, 0.00437, 0.00433, 0.00111]
q_R = [0.98943, 0.00446, 0.00504, 0.00106]
```

Sebaliknya, seed 17 dan 43 menumpukan anchor F dan R pada state berlainan.
Ini konsisten dengan **representation collapse bergantung initialization**;
ia belum membuktikan punca optimization yang tepat.

## Experiment critique

- Training mengenumerate tepat 64 state dalam float64. Ia tidak bergantung pada
  sampler THRML yang baru lulus fixture parity, jadi sampler parity tidak boleh
  menerangkan atau membaiki kegagalan seed 29.
- Finite-difference dan `lambda=0` checks menyokong consistency asas energy dan
  gradient. Tiada bukti checkpoint dipilih menggunakan test.
- Split menahan kombinasi context, bukan word type; semua anchor types muncul
  dalam train. Test retrieval menggunakan full anchor pools, jadi ia bukan
  retrieval pada perkataan baharu.
- Negatives grammar-filtered dan corruption cross-family adalah oracle-assisted
  mengikut plan. Permuted lexical control masih mengekalkan joint supervision
  dan negative rows asal; ia bukan total no-signal control.
- Random seed17 sense 0.833 tetapi AUC 0.3125, manakala permuted seed17 masih
  sense/AUC 1.0. Jadi sense toy kecil tidak boleh dijadikan bukti sendirian.

## Hypothesis critique

Prediction yang gagal ialah: konfigurasi tetap ini akan memenuhi sense,
retrieval, dan context-gain threshold bagi **setiap** seed 17/29/43.
Ini tidak membuktikan semua Ising representation gagal, tidak menguji natural
text, dan tidak menunjukkan superiority berbanding vector/Gaussian.

Competing hypothesis paling kecil ialah collapse map anchor F/R pada seed 29
menghalang context separation walaupun lexical discrimination kekal baik.

## Satu tindakan dicadangkan

Buat dan log **diagnosis validation-only, read-only** untuk tiga checkpoint
yang sudah dipilih: distribusi anchor state, per-family posterior/accuracy/gain,
region occupancy, dan ranking loss. Tiada retraining, tiada checkpoint selection
baru, dan tiada tuning test. Diagnosis itu akan membezakan anchor-map collapse
daripada contextual response yang lemah sebelum apa-apa repair model dicadangkan.
