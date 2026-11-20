> Kemas kini: bukti real-valued exp/log dan mapping formula kini siap. Lihat `LEAN_REAL_FOUR_SPIN_VERIFICATION.md`; nota di bawah merekod tahap sebelumnya.

# Lean: identiti covariance untuk konteks bersama

20 November 2026. Semakan matematik untuk Word Energy 4-spin. Ini bukti satu identiti algebra terhingga atas nombor nisbah (`Rat`), bukan bukti lengkap formula skor real-valued atau pelaksanaan JAX.

## Claim dan pemetaan kepada model

Untuk senarai keadaan pusat xs, berat q yang jumlahnya 1, serta dua fungsi mesej A dan B, takrifkan E[f] = sum_s q(s) f(s). Theorem `WordEnergy.finite_covariance_identity` membuktikan:

```text
E[(A - E[A]) (B - E[B])] = E[A B] - E[A] E[B].
```

Pernyataan Lean meliputi sebarang senarai terhingga dan fungsi bernilai Rat, termasuk senarai 16 keadaan; ia tidak memerlukan q positif untuk identiti algebra, walaupun tafsiran kebarangkalian memerlukannya. Senarai dibenarkan ada pendua; dalam aplikasi setiap konfigurasi pusat mesti disenaraikan sekali. Tiada hipotesis yang mengandaikan identiti dalam kesimpulan.

Dalam `word_energy/wikitext_four_spin.py`, q ialah `center`, `d1` dan `d2` ialah purata `expm1` bagi setiap konteks; tetapkan A=1+d1 dan B=1+d2. Oleh kerana q dinormalisasi, joint input kepada `log1p` bersamaan E[AB]-1 dan kedua-dua input pair bersamaan E[A]-1 dan E[B]-1. Sementara itu, score models menambah offset pusat sama. Pada parameter dan candidate yang sama, matematik real dan monotonic log membayangkan tanda beza log-joint dan jumlah log-pair sama dengan tanda covariance, jika E[AB], E[A] dan E[B] positif. Pemetaan exp/log, positivity, monotonicity, dan kesamaan program real/float ini belum diformalkan dalam Lean. Dua model yang dilatih berasingan tidak berkongsi parameter, jadi identiti ini tidak meramalkan AUC mereka.

## Penilaian theorem

- Root `projects/word-energy/theory`, `leanprover/lean4:v4.36.0`, Lean commit 611c117bfaf7b7ae1cfcf246ce6fba31e8f509e9; Lean core sahaja, Mathlib tiada.
- `lake build WordEnergy.Covariance`: lulus selepas sumber akhir.
- `lake env lean CovarianceAudit.lean`: memaparkan declaration dan axiom dependencies untuk expansion, covariance dan tiga fixture. Setiap satunya menggunakan `propext`, `Classical.choice`, `Quot.sound`; tiada `sorryAx` atau axiom projek. Ketiga-tiga ialah dependencies asas Lean yang lazim; tiada dasar trust projek yang mengecualikannya dikenal pasti. Tiada native decision digunakan.
- SHA-256 `WordEnergy/Covariance.lean`: `84C1AC910A1FCE1B7BF4FFD105F1DBED9A2B41DCFC7297CBDBD5D1F9DA448815`.
- Fixture Lean rasional dua keadaan mempunyai q=[1/2,1/2], A=[1,3], B=[1,3] atau [3,1]. Berat dinormalisasi; joint-minus-pair pada skala partition masing-masing +1 dan -1. Ini contoh matematik, bukan claim setiap mesej tersebut berasal daripada model spin sebenar.

## Fixture model sebenar dan batas

Perintah dari projek Word Energy: `.venv/Scripts/python.exe scripts/audit_four_spin_covariance_fixture.py`. Ia menjumlahkan 4,096 konfigurasi secara bebas, membandingkannya dengan mesej terfaktor serta scorer JAX pada parameter sama: context searah covariance +0.0776814664 dan beza log +0.0407934991; context bertentangan covariance -0.0776814664 dan beza log -0.0425286420. Kedua-duanya sepadan dalam toleransi 1e-12. Fixture ini menggunakan CPU dan floating point; ia hanya menyemak dua input.

Bukti ini menjelaskan mengapa interaksi berkongsi boleh menaikkan atau menurunkan skor per window, bukan bahawa joint mempunyai prestasi prediksi lebih baik. Ia tidak membuktikan training, generalisasi semantik, mixing THRML atau hardware. Untuk theorem lengkap model atas real, perlu real-valued finite sums, sifat exp/log dan pemetaan source-to-formula yang disahkan secara berasingan.

## Sambungan: satu mesej tidak berubah mengikut keadaan

Theorem `WordEnergy.constant_message_zero_covariance` membuktikan bagi senarai terhingga dan fungsi Rat dengan berat yang jumlahnya satu: jika mesej A adalah constant pada semua keadaan pusat, covariance A dan B adalah sifar untuk mana-mana B. Maksudnya, mesej yang tidak membezakan keadaan pusat tidak boleh menghasilkan kesan covariance bersama. Ini tidak bermaksud model mesti menghasilkan skor yang baik atau tiada bentuk interaksi lain.

`lake build WordEnergy.ConstantMessage` lulus; `lake env lean ConstantMessageAudit.lean` memeriksa declaration dan dependencies untuk theorem serta dua lemma. Ketiga-tiganya melaporkan `propext`, `Classical.choice`, `Quot.sound`, tanpa `sorryAx`. SHA-256 `WordEnergy/ConstantMessage.lean`: `0BF086704639176E532FDC9AF5A85D6A2C6BCC05A6ABC0C9DA1B4C7CCAA45ED2`. Ini theorem Rat; hubungan penuh kepada zero coupling, exp/log dan real-valued code masih belum dibuktikan secara Lean.
