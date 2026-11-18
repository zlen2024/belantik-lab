# Text8: validation baharu, tiga acakan dan vector dengan prior sama

18 November 2026. Susulan development dipilih selepas hasil sebelumnya. Tiga model berkongsi train pairs, noise, prior `0.25 log(count_train(candidate))`, loss dan budget latihan. Validation baharu daripada byte [17M,17.5M); reserved [18M,20M) tidak disampel atau dinilai. Archive checksum membaca keseluruhan fail untuk integrity sahaja. Lihat [protokol](TEXT8_FRESH_MULTI_SHUFFLE_PROTOCOL.md).

## Hasil epoch3

| Model | Parameter | AUC asal | MRR asal | AUC acak, purata 3×3 | AUC asal − acak | AUC asal − acakan baharu |
|---|---:|---:|---:|---:|---:|---:|
| tied-frequency | 3002 | 0.68201 | 0.30960 | 0.64053 | +0.04147 | +0.04213 |
| role-frequency | 6002 | 0.68434 | 0.32221 | 0.63770 | +0.04664 | +0.04710 |
| vector-frequency | 4000 | 0.68769 | 0.32576 | 0.64253 | +0.04515 | +0.04588 |

Frequency-only AUC **0.64652**, MRR **0.25830**. Semua tiga model asal mengatasi frequency-only dalam setiap seed. Acakan 11330 pernah diperhatikan dalam validation lama; 11331 dan 11332 ialah acakan baharu. Enam checkpoint Ising asal dan enam Ising 11330 digunakan semula; 24 run lain dilatih baharu. Sebahagian perbezaan asal−acak boleh mencerminkan model acak yang merosot di bawah frequency-only, bukan hanya maklumat konteks yang berguna.

Pada data dan budget ini, AUC vector asal purata 0.68769, role Ising 0.68434 dan tied Ising 0.68201. Perbezaan role−vector mengikut seed: +0.00027, −0.00268, −0.00762; tied−vector: +0.00238, −0.01163, −0.00779. Vector mempunyai 4,000 parameter berbanding 3,002/6,002 bagi Ising, maka ini bukan bukti kelebihan seni bina yang dikawal kapasiti. MRR juga diukur dan perlu dilihat bersama AUC.

## Perbezaan AUC berpasangan: asal − acak

Setiap baris ialah satu seed model; columns ialah seed permutasi. Original yang sama dikira semula bagi tiga lajur: 9 nilai bukan 9 corpus bebas. Kriteria deskriptif pra-run: semua positif dan mean >=0.005; bukan ujian statistik.

| Model | Seed | 11330 (lama) | 11331 (baharu) | 11332 (baharu) |
|---|---:|---:|---:|---:|
| tied-frequency | 17 | +0.04065 | +0.04450 | +0.04413 |
| tied-frequency | 29 | +0.04052 | +0.03781 | +0.03989 |
| tied-frequency | 43 | +0.03935 | +0.04222 | +0.04420 |
| **tied-frequency purata** | — | — | — | **semua +0.04147; baharu +0.04213** |
| role-frequency | 17 | +0.04169 | +0.04695 | +0.04515 |
| role-frequency | 29 | +0.05197 | +0.04748 | +0.04993 |
| role-frequency | 43 | +0.04351 | +0.04861 | +0.04447 |
| **role-frequency purata** | — | — | — | **semua +0.04664; baharu +0.04710** |
| vector-frequency | 17 | +0.03663 | +0.04077 | +0.04045 |
| vector-frequency | 29 | +0.04668 | +0.04720 | +0.04776 |
| vector-frequency | 43 | +0.04777 | +0.04749 | +0.05161 |
| **vector-frequency purata** | — | — | — | **semua +0.04515; baharu +0.04588** |

Ketiga-tiga model lulus kriteria deskriptif, tetapi semua berkongsi data dan validation.

## Kekerapan hampir sama

Subset comparison positive/noise distinct-ID dengan ratio count dalam [0.5,2] mempunyai **18525 comparisons**; frequency-only AUC 0.52327. Ini bukan 18,525 dokumen bebas, dan ratio masih membenarkan perbezaan frekuensi.

| Model | AUC asal | AUC acak purata |
|---|---:|---:|
| tied-frequency | 0.60412 | 0.52050 |
| role-frequency | 0.60338 | 0.51953 |
| vector-frequency | 0.60214 | 0.52013 |

## Rekod semua checkpoint

| Model | Seed | Shuffle (0 = asal) | MLflow run ID | Sumber | AUC | MRR |
|---|---:|---:|---|---|---:|---:|
| tied-frequency | 17 | 0 | `8eab021c85ea4545bef2b804dec3b716` | diguna semula | 0.68436 | 0.31092 |
| tied-frequency | 17 | 11330 | `57dd54c2e7074eb5a116188c59d4b508` | diguna semula | 0.64370 | 0.25654 |
| tied-frequency | 17 | 11331 | `42b2ee877d984bf695be52f2fd572d0a` | baharu | 0.63986 | 0.25613 |
| tied-frequency | 17 | 11332 | `3c0d031c9dd04f01bf738089c70dcd8a` | baharu | 0.64023 | 0.25583 |
| tied-frequency | 29 | 0 | `52250706518949f0bd4d8886fdf5fde5` | diguna semula | 0.67789 | 0.30811 |
| tied-frequency | 29 | 11330 | `7c1cb04b59564464a8fbf4a0d9162178` | diguna semula | 0.63736 | 0.25292 |
| tied-frequency | 29 | 11331 | `67b4ae6394164652acec46196b3b5201` | baharu | 0.64007 | 0.25573 |
| tied-frequency | 29 | 11332 | `796000cf50a94c34a05b2c08eef41015` | baharu | 0.63799 | 0.24720 |
| tied-frequency | 43 | 0 | `d56b97169f46494baac3d86ef902217a` | diguna semula | 0.68378 | 0.30979 |
| tied-frequency | 43 | 11330 | `38fba8d6fe0542158b5f06097107e044` | diguna semula | 0.64443 | 0.25740 |
| tied-frequency | 43 | 11331 | `215a8bc46c3e4623af54876bad98f748` | baharu | 0.64155 | 0.25605 |
| tied-frequency | 43 | 11332 | `c5d1def0a6244a4fa7e579b341629a5b` | baharu | 0.63958 | 0.25594 |
| role-frequency | 17 | 0 | `5a141602f79445108dc872fb0d1c3880` | diguna semula | 0.68224 | 0.32198 |
| role-frequency | 17 | 11330 | `3f9bbfd521724c6ebb1165c742467acd` | diguna semula | 0.64055 | 0.25201 |
| role-frequency | 17 | 11331 | `bedafd40a7b3405dacd9de94415d7fef` | baharu | 0.63529 | 0.24784 |
| role-frequency | 17 | 11332 | `b31c14d46af14bd0b36443d13bbb19c3` | baharu | 0.63710 | 0.25454 |
| role-frequency | 29 | 0 | `1eb62d21661049979e488a452a76777b` | diguna semula | 0.68684 | 0.32223 |
| role-frequency | 29 | 11330 | `6d59c915459d4840abc963baad30405c` | diguna semula | 0.63487 | 0.25156 |
| role-frequency | 29 | 11331 | `989e693d41cd4ac9b1e20c3e449db8bf` | baharu | 0.63936 | 0.25131 |
| role-frequency | 29 | 11332 | `e5400719f20d4bbb99d33e0202e9561a` | baharu | 0.63691 | 0.25117 |
| role-frequency | 43 | 0 | `c25f7f57ae3d4b7096de8e3468e30c01` | diguna semula | 0.68395 | 0.32243 |
| role-frequency | 43 | 11330 | `fe837ca5bfaf4258b0e9d846b1fceb9b` | diguna semula | 0.64044 | 0.25368 |
| role-frequency | 43 | 11331 | `4812514420a14be1a29f2ecd4e98d01d` | baharu | 0.63533 | 0.24976 |
| role-frequency | 43 | 11332 | `dfa39b3753a44f6ab323d29546960fdd` | baharu | 0.63948 | 0.25496 |
| vector-frequency | 17 | 0 | `753dd987fc9948c688b5575357b616d9` | baharu | 0.68197 | 0.31941 |
| vector-frequency | 17 | 11330 | `3296f10f6f9c496eb04310971fafff98` | baharu | 0.64535 | 0.25826 |
| vector-frequency | 17 | 11331 | `fdcef0f3a932418caf36641b07136b45` | baharu | 0.64120 | 0.25400 |
| vector-frequency | 17 | 11332 | `25971a89d50c4c15a540ec4e150fda75` | baharu | 0.64153 | 0.25673 |
| vector-frequency | 29 | 0 | `459303e461e845a5abf9e482f88a736d` | baharu | 0.68951 | 0.32763 |
| vector-frequency | 29 | 11330 | `1533e227749b429f9b7fe6b3730c3f80` | baharu | 0.64283 | 0.25827 |
| vector-frequency | 29 | 11331 | `0d263f15b7e94fc89adbe4d8f3b242a3` | baharu | 0.64231 | 0.25333 |
| vector-frequency | 29 | 11332 | `ae62db98dd244fd7a5165d3cf424bd21` | baharu | 0.64175 | 0.25763 |
| vector-frequency | 43 | 0 | `c4a238f7d0ad409cb68a71d5154d64c1` | baharu | 0.69156 | 0.33023 |
| vector-frequency | 43 | 11330 | `5b8baa1e34534e40b944e85a951f772d` | baharu | 0.64380 | 0.25577 |
| vector-frequency | 43 | 11331 | `e4c907c35a64418d8831f51ea0bb9c90` | baharu | 0.64407 | 0.25386 |
| vector-frequency | 43 | 11332 | `df320faf30e7410e91f0fc29e4da5972` | baharu | 0.63995 | 0.25353 |

## Data, pemeriksaan dan batas

- Validation: 5000 pasangan, 86077 token, coverage 65.02%, 20 calon noise untuk setiap pasangan. Kekerapan dan vocabulary dipelajari pada train sahaja. Noise boleh menjadi konteks sebenar; data tiada label word sense.
- Ising menggunakan exact four-spin F2; vector menggunakan directed 2D dot product. Tambahan prior digunakan semasa train dan penilaian. Zero-interaction/zero-vector menghasilkan kawalan frequency-only yang dikira pada validation baharu.
- Setiap checkpoint epoch0–3 diperiksa hash; initialization sepadan dalam setiap pasangan. Data, noise, orders dan source/hyperparameters checkpoint lama diperiksa; evaluator canonical ordered pairs mengekalkan seri ID sama. Sumber training dibekukan pada `ddf2f17`; audit skrip diperketat selepas training pada `2991335`, tanpa mengubah model atau pemilihan.
- Satu korpus text8, satu blok validation, tiga initialization dan tiga permutasi tetap: hasil bukan bukti independent-document, semantic understanding atau kelebihan thermodynamic. Tiada THRML sampler parity, ujian cip, atau tenaga dalam joule.
- Provenance byte split sejarah tidak lengkap dan text8 tiada article boundaries; split byte terpisah tidak menjamin dokumen terpisah. Tiada hyperparameter tuning pada reserved test.

MLflow experiment `word-energy-text8-fresh-multi-shuffle`: data `15facc06b0a04d978cad9a76f6300cbb`, summary `a8b22f62e575418e971aa3feb7f38bf0`. Summary menyimpan hash semua array dan source, serta semua metric per checkpoint; run latihan mempunyai checkpoint epoch dan konfigurasi. Data dan checkpoint MLflow kekal tempatan; kod dan laporan dalam Git.

Semakan bebas dan had tafsiran: [laporan critic](TEXT8_FRESH_MULTI_SHUFFLE_CRITIC.md).
