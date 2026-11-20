# Keputusan sebelum scaling WikiText-2

Gunakan checkpoint joint4 epoch 3 daripada seed 17, 29 dan 43 pada dataset f1b7db5231ed4a5788b566e388f8e731. Bandingkan pembacaan joint dengan jumlah pasangan pada checkpoint yang sama. Kekalkan perkataan, parameter, prior, 20 calon noise dan susunan konteks. Hipotesis ialah joint menaikkan kedudukan perkataan sebenar. Ukuran utama ialah beza macro AUC mengikut artikel. Kira 95% bootstrap berpasangan merentasi artikel selepas mempuratakan tiga seed (1,000 draws, RNG 23140). Rekod MRR dan bilangan perbandingan bertambah baik atau buruk, ties bernilai setengah.

Ini diagnosis development selepas eksperimen awal. Jika beza positif pada ketiga-tiga seed dan had bawah interval positif, pertimbangkan satu ujian saiz data terkawal sebelum latihan besar. Jika tidak, semak objektif dan readout. Skor vector10 terlatih kekal penanda aras untuk keputusan scaling. Set ujian reserved tidak digunakan.

Susulan eksploratori selepas run 3127680692bb423d8bdd780515b7fd65. Gunakan semula analisis pada tiga checkpoint pair_add4 epoch 3. Jadual latihan x readout menunjukkan sama ada keuntungan joint bergantung pada jenis latihan. Ini bukan ulangan bebas dan tidak mengubah keputusan primary.
