# Kawalan kekuatan W: hasil latihan

22 November 2026. MLflow experiment `word-energy-spin-coupling-control`, code/protokol `535be00`. Ini follow-up posthoc pada WikiText development yang sama dengan pilot `c61cdc3`, lima negative dan tiga seed. Tiada official test digunakan. Kita membandingkan checkpoint pada epoch akhir yang ditetapkan, tanpa memilih epoch terbaik selepas keputusan.

| Keadaan model | AUC seed 17 / 29 / 43 | Purata |
|---|---|---:|
| 4 spin, W asal jumlahnya boleh berubah hingga 1.5 | .64800 / .64635 / .65475 | .64970 |
| 4 spin, W tetap jumlah .8 | .65740 / .65445 / .66185 | .65790 |
| 4 spin, W relatif dilatih, jumlah .8 | .65685 / .65310 / .66300 | .65765 |
| 6 spin, W asal jumlahnya boleh berubah hingga 1.5 | .65380 / .63915 / .63985 | .64427 |
| 6 spin, W tetap jumlah .8 | .65980 / .65100 / .65335 | .65472 |
| 6 spin, W relatif dilatih, jumlah .8 | .66180 / .65260 / .65360 | .65600 |
| Baseline frequency | sama untuk semua seed | .66370 |

Kedua-dua kawalan jumlah .8 memberi AUC lebih tinggi daripada W asal pada semua enam pasangan seed/saiz dalam ukuran ini. Untuk 6 spin, W relatif boleh latih meningkatkan purata daripada .64427 kepada .65600; ini peningkatan **deskriptif** pada data yang sama dan intervention dipilih selepas melihat pilot. Ia masih di bawah frequency .66370 dan tidak membuktikan generalisasi atau pemahaman makna. 6 spin dengan W relatif tidak mengatasi 4 spin secara purata (.65600 berbanding .65765). W tetap ialah kawalan untuk memahami punca, bukan pengganti cadangan pengguna supaya W dilatih.

Dalam run W relatif, jumlah dikekalkan .8, tetapi agihan setiap coupling berubah: 4 spin seed17 antara .15163–.23359 daripada .2 sekata pada permulaan; 6 spin seed17 antara .09044–.15997 daripada .13333 sekata. Di sini W masih *belajar*, tetapi hanya mengagihkan kekuatan keseluruhan yang ditetapkan. Parameter unary juga dilatih. Ini ialah diagonal W positif terhad; belum arbitrary signed/dense W atau kekakuan fizikal.

Setiap run menggunakan context satu perkataan dan noise yang mungkin termasuk alternatif sah. Repeated development, data dan seed terhad, serta perubahan parameter/graph bersama bilangan spin bermakna keputusan belum membuktikan lebih spin tiada faedah. Tiada contextual-sense benchmark, Gaussian comparison, THRML atau hardware digunakan. Ujian lima kes scorer, bounds W, gradient dan input ID lulus sebelum run. Critic bebas diminta mengesahkan semua pengiraan.

Critic mencatat satu jurang rekod: protokol meminta trajectory gradient, tetapi skrip memeriksa loss dan parameter akhir finite tanpa mencatat setiap nilai gradient. Ujian gradient berasingan lulus; itu tidak menggantikan log gradient semasa semua updates. Semua 12 kawalan juga lebih baik daripada W asal pada document-macro AUC, tetapi masih di bawah baseline frequency. Lihat kritik bebas untuk interpretasi dan semakan.
