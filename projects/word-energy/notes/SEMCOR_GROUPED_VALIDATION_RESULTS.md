# SemCor grouped development baseline: hasil

22 November 2026. Protokol dan kod dibekukan dalam commit tempatan `bf734df`; MLflow run `bd4c107ad9654a1288761a207dcd7d98`. Audit awal 282 fail saringan mendapati hanya 150 fail menyumbang noun yang layak; sejumlah fail `brownv` tidak menyumbang noun. Ini bilangan fail disaring, bukan jaminan semua dokumen mengandungi contoh berguna. Pembahagian baharu menggunakan 225 fail inner-fit dan 57 inner-validation; 70 fail outer-reserved tidak dibuka.

Pilihan dua label bagi setiap perkataan hanya menggunakan inner-fit. Daripada 156 calon terpilih, 146 mempunyai sekurang-kurangnya satu contoh validation dalam dua label itu: 1,279 contoh. Sebanyak 456 contoh validation daripada sense lain bagi perkataan terpilih dikecualikan; ia tidak boleh dianggap dijawab dengan betul. Hanya 94 perkataan mempunyai kedua-dua sense terpilih dalam validation dan menyumbang kepada balanced accuracy.

| Model mudah | Accuracy per contoh | Accuracy purata per perkataan | Balanced accuracy per perkataan, 94 perkataan |
|---|---:|---:|---:|
| Pilih sense paling kerap dalam inner-fit | 0.6646 | 0.6054 | 0.5000 |
| Naive Bayes menggunakan perkataan sekitar | 0.6849 | 0.6357 | 0.5513 |

Pada contoh dan perkataan yang sama, perbezaan deskriptif ialah +0.0203, +0.0303, dan +0.0513 mengikut turutan lajur. Ini satu split development yang kecil tanpa uncertainty interval, bukan keputusan ujian buta; sense yang diketepikan, kata yang jarang, petunjuk konteks terhad, kemungkinan overlap antara korpus, pemetaan WordNet automatik dan bias pemilihan dua label mengehadkan tafsiran. Tiada perbandingan model Ising atau Gaussian, tiada pelarasan W, dan tiada pelaksanaan THRML atau hardware dalam run ini. Ia memberi alasan untuk terus membina penilaian konteks, bukan bukti bahawa Word Energy lebih baik.

Kesan W yang ingin diuji memerlukan dua model yang sama kecuali jumlah coupling `λ` tetap atau boleh dipelajari, di samping kawalan `λ=0`, baseline kekerapan, dan baseline konteks. Sebelum perbandingan itu, tetapkan pemetaan sense menggunakan inner-fit sahaja dan pastikan kaedah inference kedua-dua model menjawab pada contoh yang sama. Kekalkan 70 dokumen outer-reserved untuk protokol yang benar-benar beku.
