# Matched W strength: hasil latihan

22 November 2026. Protokol dan kod dibekukan dalam commit `190c3c4`. Enam run dalam MLflow experiment `word-energy-learnable-w-strength` tamat menggunakan data fit WikiText yang sama, model 6 spin, tiga seed dan CPU klasik. Setiap pasangan bermula dengan λ=0.8 dan parameter model sama; hanya versi learned membenarkan λ berubah. Latihan hanya menggunakan satu cue dan negative sampling, bukan label makna SemCor.

| Seed | λ akhir tetap | λ akhir dipelajari | Train loss akhir tetap | Train loss akhir dipelajari |
|---|---:|---:|---:|---:|
| 17 | 0.8000 | 1.3462 | 1.0803 | 1.0439 |
| 29 | 0.8000 | 1.3293 | 1.0844 | 1.0508 |
| 43 | 0.8000 | 1.3510 | 1.0794 | 1.0416 |

Run ID fixed/learned: seed 17 `5e1958ea3c04411bb246a1dababecf9e` / `d410f28ca0984d16ac71c4914600974d`; seed 29 `1274d6ea2d8341f1b8a972d5df7b54fb` / `10ccf64f2c544a25b983cc50ab1ea5d3`; seed 43 `da099306ecff4d4a9f5441f6d2eebcea` / `828a4c17fe694f9e9bd41f7a6c63d37f`. Ketiga-tiga learned bergerak menghampiri had parameterisasi 1.5. Ini menunjukkan optimizer memilih coupling lebih kuat untuk objective latihan ini; ia tidak menunjukkan bahawa λ ini optimum atau meningkatkan generalisasi. Train loss yang lebih rendah bukan skor pada data baharu, bukan bukti makna dan bukan perbandingan terhadap embedding sedia ada.

Gradient awal λ disemak berbanding finite difference; skor dan gradient bukan-λ pada permulaan dipadankan. Log per epoch, hash, dan checkpoint disimpan dalam MLflow. Penilaian semantik mesti dipratetap secara berasingan: fit-only sense maps untuk setiap checkpoint, contoh sama, normalisasi 2–4 cue, kawalan λ=0 dan baseline kekerapan/konteks; kekalkan outer-reserved tertutup. Belum ada penilaian Gaussian serupa, pensampelan THRML atau run fizikal.

Nota: train loss ialah purata minibatch semasa epoch terakhir, bukan penilaian checkpoint akhir pada satu set tetap.
