# SemCor: pemeriksaan berpasangan selepas keputusan

22 November 2026. Protokol dan kod disimpan dalam commit `b92e0df`; MLflow run `c20525b4457246b481e8f767d4734ec2`. Analisis post-hoc pada development yang sama, bukan test baharu dan bukan ujian Ising, Gaussian atau W. Pemilihan sense dan skor asal berjaya dibina semula: 876/1,279 tepat untuk naive Bayes berbanding 850/1,279 untuk majority, daripada 31 dokumen yang menyumbang contoh.

Beza accuracy asal ialah +0.0203. Bootstrap 2,000 kali dengan dokumen sebagai unit resampling menghasilkan julat percentile 95% dari **−0.0007 hingga +0.0439**. Julat ini termasuk sifar dan bergantung pada 31 dokumen serta andaian bootstrap; ia tidak membuktikan tiada kesan, dan tidak membuktikan peningkatan stabil.

Dalam 200 pengacakan cue bags dalam kumpulan perkataan sasaran dan bilangan cue, accuracy purata ialah **0.6665**, dengan julat percentile 95% **0.6536–0.6771**; konteks asal memperoleh **0.6849**. Sebanyak 107 contoh berada dalam kumpulan satu contoh yang cue-nya tidak boleh berpindah. Pengacakan tidak mengekalkan semua kebergantungan dokumen, maka ini kawalan deskriptif, bukan p-value. Kedua-dua analisis bersyarat pada contoh dua sense terpilih dan belum melibatkan 70 fail outer-reserved.

Perkataan sekitar mengandungi sedikit isyarat yang layak diuji lanjut, tetapi ukuran ini masih sensitif terhadap dokumen dan label terhad. Langkah projek seterusnya ialah perbandingan Ising yang mengasingkan `λ` tetap berbanding dipelajari, dengan readout, inisialisasi, data dan bajet setara; W=0 dan baseline konteks sebagai kawalan. Gaussian pada data sama dan keserasian sampler/hardware kekal kerja berasingan yang belum selesai.
