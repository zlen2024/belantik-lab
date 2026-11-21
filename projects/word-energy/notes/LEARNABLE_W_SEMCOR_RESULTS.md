# Matched W: keputusan SemCor inner development

22 November 2026. Protokol dan skrip dibekukan pada commit `aca7a2c`; MLflow run `70f04e462a774756915b9fe0951c8686`. Ini penilaian berulang pada development, menggunakan dua sense yang dipilih oleh label latihan dan centroid posterior yang juga dibentuk dengan label latihan. Sebanyak 1,279 contoh daripada 31 dokumen dinilai; ukuran balanced menggunakan 94 perkataan yang kedua-dua sense hadir. Ini bukan ujian blind atau penemuan sense tanpa label.

| Seed | W tetap pada .8 | W tetap pada kekuatan learned | W learned pada kekuatan learned |
|---|---:|---:|---:|
| 17 | 0.4826 | 0.4916 | 0.4923 |
| 29 | 0.5062 | 0.5030 | 0.4987 |
| 43 | 0.4766 | 0.4590 | 0.4808 |

Metrik ialah target-macro balanced accuracy; lebih tinggi lebih baik. W=0 memberikan 0.5000 pada setiap checkpoint kerana ramalannya tetap untuk setiap sasaran. Baseline majority juga 0.5000 dan naive Bayes konteks 0.5513 pada perkataan dan contoh terpilih yang sama. Hasil ini tidak menunjukkan W learned mengatasi baseline: lebih baik dalam satu seed berbanding W tetap asal, lebih buruk dalam satu lagi dan sedikit lebih baik pada seed ketiga, tetapi semua sekitar 0.5. Naive Bayes dan centroid mempunyai bentuk readout berbeza, maka ini perbandingan sistem, bukan bukti sebab tunggal.

Beza W learned native minus W tetap native ialah +0.0097, −0.0075, +0.0042; interval bootstrap dokumen 95% masing-masing [−0.0205, +0.0371], [−0.0324, +0.0170], [−0.0161, +0.0348]. Bootstrap melibatkan 31 dokumen dan sokongan 42–87 daripada 94 perkataan dalam setiap resample; ketidakpastian ini bersyarat dan penerokaan. Tidak ada bukti yang cukup di sini bahawa W learned memberi peningkatan yang stabil. λ sifar dipastikan menghasilkan posterior unary, dengan centroid dikekalkan pada λ native bagi setiap checkpoint.

Latihan menggunakan hanya satu perkataan konteks dan penilaian memakai 2–4 cue dengan pembahagian coupling berdasarkan bilangan cue. Objective lexical pair, geometri posterior, cara centroid dan keterbatasan data mungkin menyumbang kepada hasil lemah; eksperimen ini tidak memisahkan puncanya. Dokumen outer-reserved kekal tidak dibuka. Gaussian, pensampelan THRML dan hardware belum dinilai pada task ini.
