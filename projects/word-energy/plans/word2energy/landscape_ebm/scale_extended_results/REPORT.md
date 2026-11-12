# Susulan scaling Word2Vec-style: 5 juta sampel

Susulan ini dibuat selepas keputusan [run awal](../scale_results/REPORT.md) dilihat. Ia kajian eksploratori, bukan replika Word2Vec. Rujukan dan teknik asal diringkaskan dalam [nota kajian](../WORD2VEC_SCALING.md); rancangan susulan dalam [protokol](../SCALE_WORD2VEC_EXTENDED_PLAN.md).

## Apa yang dibuat

Kami ambil 50 juta aksara text8 (8,498,923 token sebelum subsampling) sebagai sumber latihan, 10,000 perkataan daripada train, dan 100 dimensi. Selepas subsampling kata kerap, kira-kira 1.63 juta token dikekalkan. Kami sampel **5 juta pasangan dengan penggantian**, maka bukan semuanya unik, kemudian ulang pasangan sama dua epoch: 10 juta presentation positif. Window maksimum 5, satu context per sampel, lima negatif per positif dengan taburan kekerapan kuasa 0.75. Latihan menggunakan PyTorch CPU dan Adam/SparseAdam, bukan kod SGD asal. Masa run sekitar 524 saat.

Kedua-dua model menggunakan sampel yang sama: SGNS ialah baseline dot product; neural menambah rangkaian tak linear dan terma kuadratik. Ada satu seed setiap model. Bilangan parameter juga berbeza (2,000,000 berbanding 2,038,657), maka eksperimen ini tidak mengasingkan semua kesan seni bina. Checkpoint dipilih menggunakan validation sahaja; kedua-duanya memilih epoch 2, iaitu hujung bajet. Test menggunakan aksara 85–90 juta yang belum digunakan dalam rekod eksperimen sebelumnya; validation 80–85 juta. Ini split khas, bukan split standard, dan text8 tidak menandakan sempadan dokumen. Test tidak disubsample walaupun train disubsample.

## Hasil

Setiap positif test dibandingkan dengan 20 context yang disampel daripada taburan noise. AUC rawak = 0.5 dan hit@5 rawak ≈ 0.238. Ini ukuran ranking terhadap noise, bukan accuracy makna. Ada negatif yang mungkin relevan; 0.394% negatif sama tepat dengan context positif.

| Model | AUC test | Hit@5 test | Analogi betul / 4,936 yang diliputi |
|---|---:|---:|---:|
| SGNS | 0.4897 | 0.2615 | 213 (4.32%) |
| Neural energy | 0.4838 | 0.2531 | 184 (3.73%) |

Set analogi asal mempunyai 19,544 soalan, tetapi hanya 4,936 (25.26%) boleh dinilai dengan vocab ini. Analogi menggunakan cosine bagi vektor input, bukan fungsi energy neural, dan tidak membuktikan bahawa model memahami sense. Set ini sudah dilihat sebelum susulan dirancang: keputusan analogi ialah pemerhatian post-hoc. Run awal dengan 1 juta sampel, tiga seed dan tiga epoch menghasilkan purata 0.31% dan 0.16% pada soalan yang diliputi. Susulan mempunyai lebih banyak sampel dan presentation, tetapi satu seed sahaja; ini bukan bukti bahawa bilangan sampel sahaja menyebabkan peningkatan. Split test berbeza, jadi AUC run awal dan susulan juga tidak mengasingkan kesan scaling.

## Diagnostik dan bentuk landscape

Secara post-hoc pada validation sahaja, subsampling token dan pembinaan semula window menghasilkan AUC SGNS 0.5055 → 0.6935 dan neural 0.4974 → 0.6927. Kedua-dua taburan token dan pasangan berubah sekali gus; ini konsisten dengan pengaruh perubahan taburan, tetapi tidak membuktikan sebab tunggal dan bukan skor test.

Pencarian minima numerik dilakukan selepas latihan dalam semua 100 dimensi, pada checkpoint neural seed 0. LBFGS float64 bermula dari maksimum 15 titik berdasarkan embeddings sebenar, dan calon diterima jika norma gradien < 1e-7 serta nilai eigen minimum Hessian > 1e-6. Kami menemui dua calon berbeza untuk `bank`, tiga untuk `water`, dan satu untuk `money`. Pencarian tidak menyeluruh dan bilangan minima bukan jaminan global. Untuk `bank`, dua calon terpisah sekitar 4.06 unit, tetapi setiap calon sekitar 43.7–43.8 unit daripada embedding context terdekat; median norma embedding context ialah 2.40, maksimum 6.45. Ia berada jauh dari perkataan yang dipelajari dan belum menunjukkan lembah makna sungai dan kewangan. Objektif latihan menilai titik perkataan diskret; ia tidak menjamin tingkah laku munasabah di semua titik berterusan di luar taburan. Keputusan dan koordinat ada dalam `minima_float64.json`.

Pemeriksaan contoh `bank` pada vocab mendapati `accords`, `gaza`, `departments`, dan `reserves` antara konteks energy paling rendah, manakala `river` mempunyai E=1.4938. Ini gambaran satu seed, bukan ujian semantik menyeluruh, tetapi menunjukkan ranking belum cukup baik untuk menyokong dakwaan sense. Untuk model neural run awal, carian yang sama menemui satu calon bagi setiap `bank`, `water`, dan `money` pada ketiga-tiga seed; carian tidak menyeluruh dan calon juga jauh daripada embeddings.

## Kesimpulan

Lebih banyak sampel dan presentation beriringan dengan peningkatan skor analogi dalam susulan eksploratori, tetapi prestasi masih rendah dan eksperimen belum membuktikan pemisahan dua makna `bank`. Penilaian seterusnya patut menggunakan pasangan test yang disubsample mengikut prosedur latihan selain set asal, beberapa seed dan bajet latihan lebih panjang, serta ujian konteks berlabel sense; perlu juga mengawal energy di kawasan antara dan luar embedding sebelum mentafsir minima berterusan. Tiada implementasi p-bit atau eksperimen pada cip dibuat.
