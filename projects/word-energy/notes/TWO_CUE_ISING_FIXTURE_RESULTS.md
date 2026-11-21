# Hasil fixture dua petunjuk Ising

21 November 2026. Semua parameter, keempat-empat label dan rantau pembacaan ditetapkan secara manual mengikut protokol. Target ialah landscape unary rata. Hasil ini menguji kebolehwakilan formula pada contoh yang direka, bukan kebolehan model mempelajari makna.

| Petunjuk A | Petunjuk B | Label parity | Kebarangkalian target dalam rantau `s0*s1 < 0` |
|---|---|---:|---:|
| 0 | 0 | 0 | 0.432919 |
| 0 | 1 | 1 | 0.567081 |
| 1 | 0 | 1 | 0.567081 |
| 1 | 1 | 0 | 0.432919 |

Dengan threshold 0.5, keempat-empat contoh mendapat label yang ditetapkan. Mematikan coupling menghasilkan 0.5 untuk setiap konteks. Mengira kesemua 4,096 state memberi posterior yang sama dengan rutin production sehingga ralat maksimum 4.17e-17; menukar urutan dua cue juga tidak mengubah posterior. Fixture vector yang mendarab dua tanda cue juga mencapai 4/4. Carian terhingga bagi additive threshold mendapat 3/4. Untuk bound umum, jika skor empat sel ialah c, c+u, c+v, c+u+v, kedua-dua diagonal tidak boleh negatif dan kedua-dua off-diagonal positif serentak kerana jumlah skor diagonal dan off-diagonal adalah sama.

Had utama ialah target unary **rata** dan semua internal coupling sifar. Di sini dua koordinat target dipengaruhi secara berasingan, dan sifat parity diperkenalkan oleh rantau bacaan `s0*s1 < 0` yang ditetapkan menggunakan struktur label. Ini tidak membuktikan energy landscape multi-well yang dipelajari, interaksi kolektif yang perlu untuk tugasan bahasa, atau kelebihan berbanding nonlinear vector. Model Gaussian dan pembelajaran bersama belum diuji oleh fixture ini. Formula bertindak pada CPU dan tiada pengukuran THRML atau chip.

Protokol dan kod dibekukan pada commit `9358d43` sebelum run. MLflow `word-energy-two-cue-fixtures`, run `c61f719ebfe342f49a84fad256ff11aa`, mengandungi parameter, kes, snapshot dan hasil. Critic bebas disimpan dalam nota berasingan apabila siap.
