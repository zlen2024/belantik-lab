# Ledger run MLflow — Word Energy

Snapshot ini menerangkan **setiap 78 run aktif** dalam tracker MLflow lokal pada
14 November 2026. Ia diwujudkan supaya paparan MLflow yang panjang boleh dibaca
sebagai cerita penyelidikan: soalan apa diuji, apa yang berubah, keputusan, dan
apa yang boleh atau belum boleh disimpulkan.

> Status keseluruhan: semua keputusan model bahasa di bawah ialah **synthetic
> toy / classical reference**, kecuali satu semakan parity THRML simulator pada
> fixture 6-spin. Tiada run fizikal pada thermodynamic chip, dan tiada run
> natural text berskala besar.

## Cara baca

- **ID run** ialah ID MLflow penuh. Jadi setiap run daripada UI boleh dicari di
  sini.
- Beberapa ID mengandungi hasil yang sama kerana skrip dilancar semula untuk
  log metric/artifact yang lebih lengkap. Ia tetap disenaraikan, tetapi ditanda
  **duplicate/re-log** supaya tidak kelihatan seperti bukti saintifik baharu.
- `F` dan `R` ialah dua konteks sintetik bagi dua sense `bank` (finance dan
  river). `M` ialah distractor. `1.0` sense accuracy di sini bermaksud 12/12
  window toy betul (6 F + 6 R), bukannya model sudah memahami bahasa umum.
- `lambda=0` mematikan context coupling pada checkpoint yang sama; `lambda=1`
  menghidupkannya. Perbezaannya menguji sama ada context mengubah landscape.

## Peta ringkas eksperimen

| Experiment MLflow | Soalan mudah | Keputusan ringkas |
|---|---|---|
| `infrastructure-smoke` | Bolehkah MLflow lokal simpan run dan artifact? | Ya; bukan keputusan model. |
| `reference-foundation` | Adakah formula Gaussian/Ising dan sampler simulator sepadan dengan jawapan exact kecil? | Formula/fixture disahkan; parity THRML ralat sekitar 0.0004–0.0054 pada marginal yang diuji. |
| `exact-training` | Bolehkah pilot Ising kecil belajar sense daripada toy windows? | Tidak stabil: seed 17 dan 43 lulus, seed 29 gagal; belum lulus gate tiga-seed. |
| `gaussian-lexical` | Bolehkah lexical Gaussian sahaja asingkan pasangan berkaitan? | AUC test 1.0 untuk ketiga-tiga seed pada toy. |
| `gaussian-joint-quadrature` | Adakah joint continuous landscape menggunakan context dengan kiraan numerical yang betul? | Ya pada toy: 12/12 pada tiga seed; tetapi tiada ketepatan sense melebihi frozen lexical control. Sampler klasik baharu lulus audit awal dua window. |
| `gaussian-capacity-ablation` | Adakah lebih banyak mixture components sendiri menyelesaikan context? | Tidak stabil; capacity sahaja bukan jawapan. |
| `contextual-vector` | Perlukah energy landscape untuk toy contextual readout ini? | Tidak: matched vector control juga 12/12 untuk semua seed. |

---

## 1. `word-energy-infrastructure-smoke`

| ID run | Tujuan / hipotesis | Pemboleh ubah | Hasil | Kesimpulan |
|---|---|---|---|---|
| `e6d7a2dc9e594a98b6fd2436a9851ded` — `synthetic-offline-smoke` | Uji pemasangan MLflow lokal: metric dan artifact boleh disimpan serta dibaca semula. | Bukan model; nilai sintetik `0.25`. | Run selesai dan metric boleh disenaraikan. | Infrastruktur tracker berfungsi. Ia **bukan** bukti Word Energy. |

## 2. `word-energy-reference-foundation`

| ID run | Tujuan / hipotesis | Pemboleh ubah | Hasil | Kesimpulan |
|---|---|---|---|---|
| `42f79bbb28f74992a4d1411594fc15e2` — `gaussian-and-ising-classical-reference` | Percubaan pertama membina reference exact untuk Gaussian dan Ising. | Setup reference awal. | Status MLflow `FAILED`; tiada metric akhir. | Rekod kegagalan setup, bukan hasil negatif teori. Diganti oleh run selepasnya. |
| `1fcdc91a662347b0b8fe11b2d09876a5` — `gaussian-and-ising-classical-reference` | Semak fixture Ising 6-spin secara enumerasi exact sebelum training/sampling. | Context finance lawan river. | P(`bank=++`) = 0.94208 dalam finance; P(`bank=--`) = 0.94208 dalam river; free energy pada `lambda=0` = 0. | Fixture memang mempunyai dua keadaan yang context pilih seperti dirancang. Ia masih parameter tangan, bukan pembelajaran bahasa. |
| `c3d1bffacc5145b2b1eba2b10cc249a5` — `ising-six-spin-thrml-simulator-parity` | Hipotesis: THRML block-Gibbs simulator meniru enumerasi 6-spin dalam ralat Monte Carlo kecil. | Exact enumeration vs THRML simulator; finance dan river. | Absolute error marginal 0.00040–0.00542. | Parity simulator pada fixture kecil adalah baik. Ini simulator, **bukan chip**. |
| `2d743ba7ece54edc8640954b8c299131` — `ising-six-spin-thrml-four-chain-audit-seed17` | Naik taraf smoke THRML kepada audit mixing empat chain tersebar. | RNG seed 17; finance dan river; 4 chain, warmup 500, 2,000 sample/chain. | Max R-hat 1.00064; min ESS 3,604; max MCSE 0.00392; worst error 0.00592 (bound 0.02). | Semua gate simulator fixture lulus. |
| `134fae32344f4cafbc8136f7263686f1` — `ising-six-spin-thrml-four-chain-audit-seed29` | Ulang audit THRML dengan RNG seed lain. | Seed 29 sahaja. | Max R-hat 1.00083; min ESS 4,005; max MCSE 0.00362; worst error 0.00242. | Semua gate lulus. |
| `b85dac0b672b4ced9758a371ef2ad955` — `ising-six-spin-thrml-four-chain-audit-seed43` | Ulang audit THRML dengan RNG seed lain. | Seed 43 sahaja. | Max R-hat 1.00180; min ESS 3,730; max MCSE 0.00396; worst error 0.00445. | Semua gate lulus. |

## 3. `word-energy-exact-training` — pilot Ising / exact enumeration

| ID run | Tujuan / hipotesis | Pemboleh ubah | Hasil | Kesimpulan |
|---|---|---|---|---|
| `581c2ab802b94b638681b3e76ce7c0de` — `ising-exact-ranking-smoke-seed17` | Smoke run: pipeline ranking Ising boleh train tanpa masalah asas. | Seed 17; metric awal sahaja. | Lexical loss 1.009; interaction weights kira-kira 0.88/0.84. | Hanya bukti pipeline berjalan; belum evaluasi sense/test penuh. |
| `00c89e82abdb4f72bf0f03d8461b1fcb` — `ising-exact-ranking-precheck-seed17` | Precheck dengan validation; pilih lexical dan joint checkpoint. | Seed 17; validation sahaja. | Lexical AUC 1.0; validation sense 1.0; joint loss 0.0815. | Seed 17 nampak menjanjikan tetapi test belum dibuka dalam run ini. |
| `0706fb0f32504442a8e8de7b52214222` — `ising-exact-ranking-precheck-seed17` | Test frozen bagi precheck seed 17. | Sama model seed 17, tambah test. | Test sense 1.0; posterior basin betul 0.9966 vs prior 0.5; retrieval AUC 1.0. | Lulus untuk satu seed sahaja. |
| `4eb16727f30c48c5bd1b8a455534f6dc` — `ising-exact-ranking-precheck-seed29` | Ulang protokol Ising pada seed 29. | Hanya seed berubah. | Validation/test sense 0.5; posterior gain hampir sifar; lexical AUC masih 1.0. | Contextual/sense bahagian gagal walaupun lexical retrieval bagus. |
| `45b5fa9ee2a34c42972e16b484ed7298` — `ising-exact-ranking-precheck-seed43` | Ulang protokol Ising pada seed 43. | Hanya seed berubah. | Test sense 1.0; gain 0.4968; lexical AUC 1.0. | Dua seed berjaya, satu gagal; belum stabil. |
| `e43c84d4baa64aec9ed442e6f1c2e636` — `random-control-seed17` | Negatif control: adakah skor tinggi muncul tanpa training? | Raw initialization, seed 17. | Sense 0.833, tetapi retrieval macro AUC 0.3125. | Sense metric kecil boleh kelihatan tinggi secara kebetulan; retrieval/control wajib dilapor. |
| `60a35feb7c8849a5a6cf4ab4d93518c9` — `random-control-seed29` | Negatif control seed 29. | Raw initialization. | Sense 0.5; AUC 0.2344. | Tiada signal stabil tanpa training. |
| `317f587c6c104e83ab92d8be1e51b0e4` — `random-control-seed43` | Negatif control seed 43. | Raw initialization. | Sense 0.5; AUC 0.6719. | AUC raw juga berbeza ikut seed; toy kecil perlu ditafsir berhati-hati. |
| `3a848dac2558421baacf5746de1162a6` — `permuted-lexical-control-seed17` | Control: rosakkan pasangan lexical positif tetapi kekalkan struktur lain. | Positive lexical context dipermutasi; seed 17. | Sense 1.0, gain 0.4943, retrieval AUC 1.0. | Untuk seed ini control masih lulus; ia melemahkan claim bahawa lexical positives sahaja menerangkan hasil. |
| `70c459afafef4b548e4ced1505134a59` — `permuted-lexical-control-seed29` | Control sama pada seed 29. | Hanya seed berubah. | Sense 0.5, gain 0.0068, AUC 1.0. | Mengulang ketidakstabilan seed 29. |
| `ebe20c3bdb0249858222eb0455d39cda` — `permuted-lexical-control-seed43` | Control sama pada seed 43. | Hanya seed berubah. | Sense 0.5, gain 0.0070, AUC 1.0. | Training/architecture belum cukup stabil untuk claim pilot Ising. |

**Kesimpulan eksperimen Ising:** precheck tiga-seed gagal gate sebab seed 29 tidak
membezakan sense. Random control dan permuted control pula menunjukkan metrik
kecil ini bukan bukti kukuh sendiri. Ia bukan keputusan “Ising mustahil”, tetapi
hipotesis pilot ini belum disokong pada protokol sekarang. Critique dan tindakan
read-only yang dicadangkan direkod dalam
[`notes/ISING_PILOT_CRITIQUE.md`](notes/ISING_PILOT_CRITIQUE.md).

Diagnosis susulan validation-only: `4ab120cbd8af48b787d482a2d129e80f`
(seed17), `aa7fdddb3d2146ec84c667f2b5e685f3` (seed29), dan
`b1683c935a044a8e97576908599e79ad` (seed43). Tujuan: membezakan collapse
anchor daripada respons context lemah; yang berubah hanya checkpoint seed.
Seed29 memperoleh F 6/6, R 0/6 dan perbezaan anchor F/R 0.000807; seed17/43
memperoleh 6/6 bagi setiap keluarga dan perbezaan anchor ~0.997. Diagnosis ini
konsisten dengan collapse, tetapi tidak membuktikan puncanya. Butiran, prior,
gain dan loss berada dalam [nota diagnosis](notes/ISING_VALIDATION_DIAGNOSIS.md).

Diagnosis tambahan random dan permuted bagi tiga seed, beserta protokol
attribution yang telah dikritik, direkod dalam
[protokol attribution](notes/ISING_COLLAPSE_ATTRIBUTION_PROTOCOL.md). Enam run
baharu membandingkan checkpoint sedia ada; bukan latihan baharu.

## 4. `word-energy-gaussian-lexical` — baseline lexical sahaja

| ID run | Tujuan / hipotesis | Pemboleh ubah | Hasil | Kesimpulan |
|---|---|---|---|---|
| `5dfae722565a4fe58dbf16837b020ae4` — `gaussian-lexical-precheck-seed17` | Uji sama ada normalized Gaussian mixture lexical boleh rank F/R lawan M. | M2, seed 17. | Validation dan test macro AUC 1.0; loss 0.0241. | Lexical toy baseline lulus. |
| `904409dee44643ec8207f874edbe02e8` — `gaussian-lexical-precheck-seed29` | Ulang seed 29. | Seed sahaja. | Validation/test AUC 1.0; loss 0.0160. | Stabil pada seed ini. |
| `ced3374c994a49f2bbe9895249060311` — `gaussian-lexical-precheck-seed43` | Ulang seed 43. | Seed sahaja. | Validation/test AUC 1.0; loss 0.0228. | Stabil pada tiga seed untuk retrieval toy. |
| `2ad251618d764db5be60ed561c20c1bf` — `gaussian-lexical-precheck-seed43` | **Duplicate/re-log** seed 43. | Tiada pemboleh ubah baharu. | Metric sama tepat seperti `ced337…`. | Salinan tracker; bukan eksperimen baharu. |

## 5. `word-energy-gaussian-joint-quadrature` — continuous joint landscape

### 5a. Training/numerical precheck dan sempadan sense

| ID run | Tujuan / hipotesis | Pemboleh ubah | Hasil | Kesimpulan |
|---|---|---|---|---|
| `ea1422fb0ebd43aa9c8bb3412aacd538` — `joint-quadrature-precheck-seed17` | Pastikan joint free-energy Gaussian boleh dikira dengan Gauss-Hermite quadrature. | Seed 17; order 8. | `F(lambda=0)` error 8.9e-16; joint loss 0.5658. | Asas numerik konsisten. |
| `bd2116ff1ba94c6cbf4a98aad69bc52a` — `joint-quadrature-precheck-seed17` | Tambah refinement check kepada precheck seed 17. | Order 8 vs refinement. | Delta free energy 2.7e-7; mean-gradient delta 2.7e-6. | Kiraan cukup stabil untuk peringkat ini. |
| `b5eb330747654a91820ab2f2b3f4cb5b` — `joint-validation-region-audit-seed17` | Audit awal region sense. | Formula region awal. | Accuracy 1.0, tetapi mass delta 0.00114. | Audit awal kemudian **diganti** apabila bug/definisi sempadan tail dibetulkan. Jangan guna sebagai hasil akhir. |
| `fb7fc512dd1d466b8c3e2bda882294cf` — `joint-validation-region-audit-seed17` | Audit region yang lebih ketat. | Adaptive integration dan domain checks. | Accuracy 1.0; max mass delta 1.05e-7. | Numerik lebih baik, tetapi tetap sebelum pembaikan penuh selector/boundary. |
| `99df11c7083944fabb947ee637628e00` — `joint-quadrature-precheck-seed17` | Precheck terkini seed 17. | Candidate eligibility dan validation sense turut direkod. | Validation sense 1.0; 5 candidate numerically eligible. | Checkpoint seed 17 layak untuk audit corrected. |
| `7c0ccc5fe0a84676bdfec71ec7e61ccf` — `joint-quadrature-precheck-seed29` | Precheck terkini seed 29. | Seed 29. | Validation sense 1.0; numerical deltas <4.5e-6. | Layak untuk audit corrected. |
| `4158173c2ea248ed8cc7ffb78a62b8bd` — `joint-quadrature-precheck-seed43` | Precheck terkini seed 43. | Seed 43. | Validation sense 0.5; numerical checks baik. | Kegagalan ini mendorong pemeriksaan geometri/boundary, bukan dianggap numerical error. |
| `f463a1e1df3f400c99fd10082f5bf4a8`, `8a1ff2ece95045a7bc3a10a35b7413ae`, `c06bb8cced5047c4b220a61ccfad319d` — `lexical-validation-region-audit` seeds 17/29/43 | Audit validation frozen lexical sebelum pembetulan terakhir. | Seed dan checkpoint lexical. | Sense 1.0/1.0/0.5; posterior gain 0.083/0.091/0.043. | Versi ini menggunakan boundary lama; diganti oleh baris selepas pembaikan. |
| `88cd054b0db74af08d9ff102def8b3c3`, `fa1fdbe58dca41768ac1e518545fbd06`, `a56b0710b8b8467f861a3eb8a0b5916c` — corrected `lexical-validation-region-audit` seeds 17/29/43 | Uji frozen lexical control dengan region crossing yang betul. | Boundary repair; tiga seed. | Semua sense 1.0; gain 0.0832/0.0912/0.0869; numerical mass delta <=4.4e-7. | Lexical control sendiri stabil dan boleh gunakan context readout dalam toy ini. |
| `0326d74e5f1a4ddeb5e4c90ea02e1eea`, `c82bdb2e77d1462ca84f35743d742274`, `e602089df9cd46cea08bc27355dd5ec2` — corrected `joint-validation-region-audit` seeds 17/29/43 | Uji joint model dengan boundary repair yang sama. | Joint vs frozen lexical; tiga seed. | Semua sense 1.0; gain 0.0870/0.0954/0.0906; numerical mass delta <=4.3e-7. | Joint lulus toy validation, tetapi perbezaan daripada lexical kecil. |
| `04761976239e4cb4bcdbcf16192a0c08` — `gaussian-validation-lexical-joint-geometry-diagnosis` | Mengapa joint dan lexical sama accuracy? | Banding geometry selepas repair. | `joint - lexical accuracy = 0` untuk semua seed; joint gain lebih tinggi hanya kira-kira 0.002–0.004. | Tiada bukti advantage joint pada metrik sense toy ini. |

### 5b. Frozen test (jangan guna untuk tuning)

| ID run | Tujuan / hipotesis | Pemboleh ubah | Hasil | Kesimpulan |
|---|---|---|---|---|
| `a68eebbf48704307b10d8a7d3f6eb499`, `d998c9c605004cf18126ae3a8894dfac`, `ea18b48484ac4ca4918a1586d14b83db` — early `lexical-test-region-audit` seeds 17/29/43 | Buka frozen test untuk lexical control. | Tiga seed; initial compact log. | Sense 1.0, gains 0.0843/0.0918/0.0877. | Hasil test awal; dilog semula dengan counts dan retrieval lengkap di bawah. |
| `47a2a8d207474bdb9681340b220bb67d`, `652696049a374f41a7a7b077ce2c6d08`, `f70148df7d724ad380e61003dfc728b7` — early `joint-test-region-audit` seeds 17/29/43 | Buka frozen test untuk joint model. | Tiga seed; initial compact log. | Sense 1.0, gains 0.0879/0.0960/0.0912. | Hasil test awal; dilog semula dengan counts dan retrieval lengkap. |
| `2d3449f813f84f959319d1ebded856e7`, `7633bf88c9094f6f853de01bec9a35af`, `11a6216b26e540acb3abe3d9562d4d83` — full `lexical-test-region-audit` seeds 17/29/43 | Rekod lengkap test lexical: sense, 6/6 F, 6/6 R, ties, retrieval. | Frozen lexical checkpoint; tiga seed. | Semua 12/12 sense, zero cross-sense error/tie, retrieval AUC 1.0, gains 0.0843/0.0918/0.0877. | Control klasik lexical lulus sepenuhnya dalam toy. |
| `5837d7df20b64ec681d718e7f83e2f3e`, `3e0052d44cf84a16a8d328de7aaf66ff`, `ae32b6a704144a4aa919b7a64373bdcc` — full `joint-test-region-audit` seeds 17/29/43 | Rekod lengkap test joint dengan metrik sama. | Joint checkpoint; tiga seed. | Semua 12/12 sense, zero error/tie, retrieval AUC 1.0, gains 0.0879/0.0960/0.0912. | Joint lulus, tetapi tidak mengatasi control pada accuracy/retrieval. |
| `e28c49114ca54c4182d0eaa5ee3b8b76`, `024d0c6df2ad4fde99014adc95492ce8` — full joint test duplicate seeds 29/43 | **Duplicate/re-log** final test. | Tiada perubahan protokol/model. | Metric sama tepat dengan `3e0052…` dan `ae32b6…`. | Salinan tracker, bukan sampel/eksperimen baharu. |

### 5c. Audit sampler classical MCMC

| ID run | Tujuan / hipotesis | Pemboleh ubah | Hasil | Kesimpulan |
|---|---|---|---|---|
| `f072e4df9f504ca99d8ac91ea22848ec` — sampler audit seed17 | Semak sampler Metropolis-Hastings joint lawan quadrature exact, bukan hanya formula. | 4 dispersed chains; burn 500; 2,000 samples/chain; validation row 0(F) dan row 6(R). | Acceptance ~0.920–0.922; R-hat ~1.000; ESS >=5873; max error 0.00454. | Lulus gate awal pada dua fixture. |
| `43e224fcc60142358e4c3fbe0cfaa0f1`, `a1c2d9f8b90240708b1c175bbf0a8808` — sampler audit seeds29/43 (first logs) | Audit sama bagi seed lain. | Seed sahaja. | Seed29 max error 0.00367; seed43 max error 0.01375; kedua-duanya memenuhi bound `max(0.02, 3×MCSE)`. | Lulus; kemudian re-log akhir dibuat untuk ID yang konsisten. |
| `b01af99bcd37480b94aad85dd7445ae7`, `3307ecad9c7846e3a9df93c2c6b737a8` — sampler audit seeds29/43 (final re-log) | **Duplicate/re-log** audit seed29/43. | Tiada perubahan sampler/protokol. | Nilai sama dengan dua ID di atas. | Bukti aktif setakat ini: tiga seed lulus pada **dua** validation window sahaja. Belum audit semua window atau THRML/Torx sampler path. |

**Kesimpulan Gaussian joint:** pembaikan sempadan menjadikan pengukuran
numerically stable. Pada frozen synthetic test, joint dan lexical sama-sama
sempurna; joint cuma menaikkan posterior gain sedikit. Jadi ia membuktikan
mekanisme toy boleh berfungsi, **bukan** bahawa energy embedding lebih baik
daripada vector/lexical embedding. MCMC classical lulus audit awal, tetapi
perlu audit seluruh validation window sebelum menjadi rujukan inference yang
lebih luas.

## 6. `word-energy-gaussian-capacity-ablation`

| ID run | Tujuan / hipotesis | Pemboleh ubah | Hasil | Kesimpulan |
|---|---|---|---|---|
| `6645915809fc4413bd9b3ee4d9a4797c`, `4217bc84e0dc446aac5d358a185ccbed`, `2e07f67b01b34df4851a9d15dd411bfa` — `capacity-m1` seeds 17/29/43 | Uji jika satu mixture component (M1) mencukupi untuk context. | Capacity M=1; seed berubah. | Sense 0.5/0.5/1.0; posterior gain 0.0735/0.0367/0.0908. Seed17 juga mempunyai domain delta 0.356, jadi audit itu tidak reliable. | M1 tidak stabil; tidak cukup bukti satu lembah memadai. |
| `725c43fd55694e6aa5d36c5759255a07`, `d909cb90db814b689c293ec264475e26`, `f5b6ee907a2c42ecba0c2380c9538f52` — `capacity-m4` seeds 17/29/43 | Uji jika empat components (M4) automatik memperbaiki sense. | Capacity M=4; seed berubah. | Sense 1.0/0.5/0.5; gain 0.1104/0.0413/0.0472. | M4 juga tidak stabil. Lebih banyak mode bukan penyelesaian sendiri. |
| `a8d823aedd23454aad0e1976b6f8daf5`, `d63256959d2f42638fd4cefa794d35e4` — M4 duplicates seeds 29/43 | **Duplicate/re-log** M4. | Tiada perubahan. | Metric sama tepat dengan M4 seed29/43 di atas. | Bukan bukti tambahan. |

## 7. `word-energy-contextual-vector` — matched ordinary vector control

| ID run | Tujuan / hipotesis | Pemboleh ubah | Hasil | Kesimpulan |
|---|---|---|---|---|
| `b7da3ffa0235427cbf538ebfe10df030`, `7e5ae3c6180d43e3a3f191e32c63fab4`, `7e48b0c0ee754eca90967c6c75d72603` — tied D6 validation seeds 17/29/43 | Matched parameter-budget vector baseline; bolehkah input/output table yang sama baca context? | Tied SGNS, D=6, 156 parameters. | Semua validation 12/12; zero error/tie. | Vector biasa lulus toy. |
| `51bff68f104947f1a92ca50d289d1e2d`, `317dee63ee9f444292cb41f2c87f1947`, `c76760a195184033bce68d81c2c4c925` — untied D3 validation seeds 17/29/43 | Control parameter-budget sama dengan dua table berasingan. | Untied SGNS, D=3 + D=3, juga 156 parameters. | Semua validation 12/12; zero error/tie. | Keputusan tidak bergantung pada tied/untied dalam toy. |
| `66d8c9240558495cb4af06ccfe10df030`, `40a2200734a34991896a57a3e38dce5c`, `dbfccbf0b0ba4cb4802165b0634f8153` — tied D6 test seeds 17/29/43 | Frozen test vector tied. | Model frozen; tiga seed. | Semua test 12/12; zero error/tie. | Tiada kelebihan Gaussian joint pada task ini. |
| `fd25d54c19034a398b83c309936d9002`, `d2c0874a28884e39aa88ccc84fd0fd78`, `1e79b9b1a5404c518bcbc3eb3586ffb3` — untied D3 test seeds 17/29/43 | Frozen test vector untied. | Model frozen; tiga seed. | Semua test 12/12; zero error/tie. | Control kedua juga sempurna. |
| `156ad9e9ff0d4836a49125ec6beda409`, `461679edafa3447baecea3f0b3ada663`, `dfc998c3ceb340a5ba789d580821fe5c` — untied D3 test duplicates | **Duplicate/re-log** test untied. | Tiada perubahan model/protokol. | Metric sama dengan tiga test untied di atas. | Bukan bukti tambahan. |

> **Pembetulan ID:** tiga tied D6 test IDs dalam tracker ialah `66d8c924…`,
> `40a22007…`, dan `dbfccbf0…`. Baris pertama di atas sengaja merujuk run
> `66d8c9240558495cb4af06ccfe10df030` seperti direkod MLflow.

## Kesimpulan merentas semua run

1. **Kita sudah ada alat ukur yang lebih jujur.** Exact enumeration/quadrature,
   boundary repair, frozen holdout, control vector, dan sampler diagnostics
   memisahkan “graf nampak cantik” daripada evidence sebenar.
2. **Gaussian landscape dan joint context berfungsi pada toy**, dan MCMC
   classical awal sepadan dengan exact reference. Tetapi joint tidak memberi
   accuracy/retrieval advantage berbanding frozen lexical control; vector
   control juga sempurna.
3. **Pilot Ising masih belum boleh dinaikkan taraf.** Satu daripada tiga seed
   gagal dan controls menunjukkan toy metric mudah dipengaruhi initialization.
4. **Sampler classical kini lulus gate tiga-seed pada semua 18 validation
   window** (tambahan bertarikh 15 November di bawah). Langkah berikutnya ialah
   menerjemah fixture/observable sama ke THRML atau Torx dan mengukur parity
   terhadap reference classical; jangan tune lagi pada frozen test.

## Lokasi bukti

- Tracker metadata: `tracking/mlflow.db`. Semak `artifact_uri` run sebelum backup:
  run semasa menggunakan `mlruns/<experiment>/<run>/artifacts/`, bukannya
  `tracking/artifacts/`. Kedua-dua lokasi lokal yang dirujuk oleh tracker perlu
  dibackup bersama; jangan andaikan satu root artifact untuk semua run lama.
- Frozen Gaussian protocol:
  [`plans/word2energy/M2_CONTEXTUAL_HOLDOUT_MANIFEST.md`](plans/word2energy/M2_CONTEXTUAL_HOLDOUT_MANIFEST.md).
- Frozen vector control:
  [`plans/word2energy/CONTEXTUAL_VECTOR_HOLDOUT_MANIFEST.md`](plans/word2energy/CONTEXTUAL_VECTOR_HOLDOUT_MANIFEST.md).
- Ising/Z1 plan dan batas claim hardware:
  [`plans/word2energy/WORD_ENERGY_Z1_FIRST_PLAN.md`](plans/word2energy/WORD_ENERGY_Z1_FIRST_PLAN.md).

---

## Addendum 15 November 2026 — sampler audit seluruh validation split

Semasa memperluas audit dua-fixture, run batch awal merangkumi row `0–11`
sahaja: semua enam F dan enam R. Semakan gabungan mendedahkan validation split
sebenar mempunyai **18 row** (`0–17`), dengan enam distractor M pada `12–17`.
Ini ialah *coverage error* pada laporan awal, bukan perubahan checkpoint atau
protocol. Artifact separa tidak dipadam. Enam row M kemudian diaudit dengan
jadual sama: empat chain tersebar, burn-in 500, 2,000 retained sweeps/chain,
seed deterministik mengikut row, quadrature order 12.

Critic bebas menyemak kod/data dan mengesahkan audit membaca hanya
`validation_windows` serta sense map daripada train anchors; ia tidak menggunakan
test untuk memilih checkpoint atau metrik sampler. Perincian pembetulan dan
had claim ada dalam
[`notes/SAMPLER_AUDIT_COVERAGE_CORRECTION.md`](notes/SAMPLER_AUDIT_COVERAGE_CORRECTION.md).

| ID run | Tujuan / hipotesis | Pemboleh ubah | Hasil | Kesimpulan |
|---|---|---|---|---|
| `486fec85383b4568b4f8967b9b96a4b1`, `9bebb825b23b4d7b9443009cd1c803b1`, `96e23d346b5744afb36b3d109af4fe6c` — seed17 row 0–1, 2–3, 4–5 | Rekod batch F pertama sambil mengelak had masa proses. | Row validation sahaja; schedule tidak berubah. | Artifact meliputi row 0–5. | Bahan bukti separa, kemudian digabungkan. |
| `c081789041884b52acfdd39f4d007f23`, `bce36765dec1462987c7c994bb49c962` — seed17 row 6–7; `796f0e9cc7664008b845a2fcf55e4a10`, `c8855934a5b944e980b646d24b63abd1` — row 8–9; `883d5442b49749b1b32b11cb113c521a`, `03335c86b164409fa36cf6dca3d19258` — row 10–11 | Rekod batch R seed17. | Sama; sesetengah batch re-log selepas proses UI tamat lewat. | Duplicate dalam setiap pasangan mempunyai metric/artifact sama. | Duplicate ialah rekod tracker, bukan run saintifik baharu. |
| `bc5f465f9ca1418899799be6ef27dfdb` — seed17 `all-validation` | Gabung dan nilai kesemua 18 row selepas M ditambah. | 18 row F/R/M; 4×(500 + 2,000). | Acceptance 0.9160–0.9394; max R-hat 1.000668; min ESS 5,292.8; max MCSE 0.00656; worst error 0.01092 (row7 finance basin, bound 0.02). | Semua gate lulus. |
| `1a666423fd884b95ba84d7b1fea8ecec` — seed29 `all-validation` | Ulang split-wide audit seed29. | Seed sahaja. | Acceptance 0.9137–0.9391; max R-hat 1.000829; min ESS 4,981.3; max MCSE 0.00671; worst error 0.01217 (row7 finance basin, bound 0.02). | Semua gate lulus. |
| `7806a4c8ef4d4745a99cc4e115350b42` — seed43 `all-validation` | Ulang split-wide audit seed43. | Seed sahaja. | Acceptance 0.9167–0.9388; max R-hat 1.000718; min ESS 5,003.8; max MCSE 0.00703; worst error 0.01375 (row6 finance basin, bound 0.02). | Semua gate lulus. |

**Kesimpulan addendum:** classical independence-MH sampler kini melepasi gate
yang dipersetujui pada tiga seed dan semua 18 validation row untuk checkpoint
Gaussian joint M2 ini. Ini mengesahkan sampler vs quadrature pada toy tersebut;
ia tidak menunjukkan semantic advantage, natural-text generalization, THRML/Torx
parity, atau execution pada physical thermodynamic hardware.
